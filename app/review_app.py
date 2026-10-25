"""Staff review app: AI suggestions in, staff decisions out (accept, edit, reject, defer).

Run (from the repo root):
    python -m greenroom run --all
    python -m streamlit run app/review_app.py --server.address 127.0.0.1

Reads <data>/records/<video_id>/records.jsonld, plays the window clip, shows
the fused suggestion with its belief, disbelief and uncertainty and the
per-channel evidence behind it, and appends every decision to the
hash-chained ledger <data>/records/<video_id>/reviews.jsonl. Nothing here
asserts anything about a participant: the five expression words are a
staff-only field that stays empty unless a staff member chooses one.
"""
from __future__ import annotations

import json
import time

import pandas as pd
import streamlit as st

from greenroom import config, ledger, uploads
from greenroom import taxonomy as T
from greenroom.expression_timeline import build_expression_timeline, expression_timeline_csv
from greenroom.media import cut_subclip
from greenroom.records import load_records
from greenroom.windows import load_index, windows_for_video

st.set_page_config(page_title="greenroom | staff review", layout="wide")

CONTEXT_PAD_S = 5.0
QUEUES: dict[str, set[str] | None] = {
    "Review required": {"staff_review_required"},
    "Mixed activity": {"mixed_activity"},
    "Confident suggestions": {"ai_suggested"},
    "Everything": None,
}
STATUS_TEXT = {
    "staff_review_required": "Staff review required",
    "mixed_activity": "Mixed activity suggested",
    "ai_suggested": "AI suggestion (unverified)",
}


@st.cache_data(show_spinner=False)
def _index(path: str, modified_ns: int) -> dict:
    return load_index(path)


@st.fragment(run_every="3s")
def _upload_panel(app_run: int) -> None:
    """Upload controls and durable job status refresh without interrupting review."""
    fragment_only = st.session_state.get("upload_panel_app_run") == app_run
    st.session_state["upload_panel_app_run"] = app_run
    with st.expander("Add videos", expanded=False):
        st.caption("Choose videos to analyze and add to the review list. Uploads and processing stay on this computer.")
        generation = st.session_state.get("upload_generation", 0)
        selected = st.file_uploader(
            "Choose videos",
            type=sorted(ext.lstrip(".") for ext in config.VIDEO_EXTENSIONS),
            accept_multiple_files=True,
            key=f"video_uploads_{generation}",
            help="Up to 2 GB per video. Existing videos and reviews are kept.",
        )
        if st.button("Add to processing queue", disabled=not selected, key="queue_uploads"):
            notices = []
            with st.spinner("Checking and saving videos…"):
                for uploaded in selected:
                    try:
                        job = uploads.submit_upload(uploaded)
                        notices.append(("success", f"{job['original_name']}: added to the processing queue."
                                        if job["status"] == "queued" else f"{job['original_name']}: already {job['status']}."))
                    except Exception as exc:  # noqa: BLE001
                        notices.append(("error", f"{uploaded.name}: {exc}"))
            st.session_state["upload_notices"] = notices
            st.session_state["upload_generation"] = generation + 1
            if fragment_only:
                st.rerun()
        for level, message in st.session_state.get("upload_notices", []):
            (st.success if level == "success" else st.error)(message)

    try:
        jobs = uploads.list_jobs()
        if any(job["status"] == "queued" for job in jobs):
            uploads.start_worker()
    except Exception as exc:  # noqa: BLE001
        st.session_state["upload_queue_error"] = str(exc)
        st.error(f"Could not start or read video processing: {exc}")
        return
    st.session_state["upload_jobs"] = jobs
    st.session_state["upload_queue_error"] = None

    completed = {job["job_id"] for job in jobs if job["status"] == "completed"}
    previous = st.session_state.get("completed_upload_jobs", completed)
    st.session_state["completed_upload_jobs"] = completed
    if completed - previous and fragment_only:
        # A full app run already reloads the index below. Interrupting it here would
        # discard review widgets that have not rendered yet, including unsaved input.
        st.rerun()
    if jobs:
        with st.expander("Upload progress", expanded=any(job["status"] in {"queued", "running", "failed"} for job in jobs)):
            # Always retain unfinished jobs and their retry controls; cap only completed history.
            visible_jobs = [job for job in jobs if job["status"] != "completed"]
            visible_jobs += [job for job in jobs if job["status"] == "completed"][:10]
            for job in visible_jobs:
                st.text(job["original_name"])
                status = job["status"]
                if status == "completed":
                    st.caption(f"Ready to review · {job['n_windows']} windows")
                elif status == "running":
                    st.caption("Processing locally… You can keep reviewing other videos.")
                elif status == "queued":
                    st.caption("Queued for processing")
                else:
                    st.error(job.get("error") or "Processing stopped before this video was ready.")
                    if st.button("Retry processing", key=f"retry_upload_{job['job_id']}"):
                        try:
                            uploads.retry_job(job["job_id"])
                        except Exception as exc:  # noqa: BLE001
                            st.error(str(exc))
                        else:
                            uploads.start_worker()
                            if fragment_only:
                                st.rerun()


def _name(lid: str | None) -> str:
    return T.DISPLAY_NAME.get(lid, lid or "")


def _subclip(window: dict, source_file: str, context: bool) -> str:
    video_path = config.videos_dir() / source_file
    folder = config.subclips_dir() / window["video_id"]
    if context:
        start = max(0.0, window["start_s"] - CONTEXT_PAD_S)
        duration = (window["end_s"] - window["start_s"]) + 2 * CONTEXT_PAD_S
        dst = folder / f"ctx_{window['index']:05d}.mp4"
    else:
        start, duration, dst = window["start_s"], window["end_s"] - window["start_s"], folder / f"{window['index']:05d}.mp4"
    return str(cut_subclip(video_path, dst, start, duration))


def _first_pending(queue: list[dict], decided: dict, start_at: int = 0) -> int | None:
    order = list(range(start_at, len(queue))) + list(range(0, min(start_at, len(queue))))
    for i in order:
        if queue[i]["window_id"] not in decided:
            return i
    return None


def _label_table(record: dict, top_n: int = 5) -> pd.DataFrame:
    rows = []
    for lid, op in record["opinions_by_label"].items():
        rows.append({"label": _name(lid), "P": op["P"], "belief": op["b"], "disbelief": op["d"], "uncertainty": op["u"]})
    df = pd.DataFrame(rows).sort_values("P", ascending=False).head(top_n)
    return df.reset_index(drop=True)


def _channel_table(record: dict, lid: str) -> pd.DataFrame:
    rows = []
    for channel, ops in record["opinions_by_modality"].items():
        op = ops.get(lid)
        if op:
            rows.append({"channel": channel, "P": op["P"], "belief": op["b"], "disbelief": op["d"], "uncertainty": op["u"]})
        else:
            rows.append({"channel": channel, "P": None, "belief": None, "disbelief": None, "uncertainty": 1.0})
    return pd.DataFrame(rows)


def _merged_export(records: list[dict], decided: dict) -> list[dict]:
    merged = []
    for r in records:
        d = decided.get(r["window_id"])
        out = {k: v for k, v in r.items() if k not in ("opinions_by_label", "opinions_by_modality")}
        if d:
            out.update(
                {
                    "reviewer_decision": d["decision"],
                    "primary_activity": d["final_primary"] or r["primary_activity"],
                    "secondary_activities": d["final_secondary"] if d["final_primary"] else r["secondary_activities"],
                    "facilitation_flag": d["facilitation_flag"],
                    "staff_contextual_observation": d["staff_contextual_observation"],
                    "expression_label_staff": d["expression_label"],
                    "reviewer_note": d["note"] or None,
                    "review_date": d["timestamp"],
                    "reviewer": d["reviewer"],
                    "origin": "staff_reviewed",
                    "ledger_hash": d["hash"],
                }
            )
        merged.append(out)
    return merged


def _expression_timeline(video_id: str, video: dict, records: list[dict], decided: dict, windows: dict) -> None:
    document = build_expression_timeline(video_id, records, decided, video["source_file"])
    entries = document["windows"]
    st.subheader("Timestamped expression tags")
    st.caption(
        "Times refer to the original video. AI suggestions describe visible expressions, not how anyone feels. "
        "Staff observations are recorded separately."
    )
    tagged = sum(entry["ai_expression"] is not None for entry in entries)
    st.write(f"{tagged}/{len(entries)} windows have AI expression suggestions.")
    selected_labels = st.multiselect(
        "Filter expression tags",
        ["Positive", "Neutral", "Negative", "Upset", "Confident", "No AI suggestion"],
        key=f"timeline_labels_{video_id}",
        help="Matches either an AI suggestion or a staff observation. Leave empty to show all windows.",
    )
    visible = [entry for entry in entries if not selected_labels or
               (entry["ai_expression"] or {}).get("label", "No AI suggestion") in selected_labels or
               (entry["staff_expression"] or {}).get("label") in selected_labels]
    if visible:
        rows = []
        for entry in visible:
            ai = entry["ai_expression"] or {}
            staff = entry["staff_expression"] or {}
            rows.append({
                "Start": entry["start_time"], "End": entry["end_time"],
                "AI expression (unverified)": ai.get("label", "No AI suggestion"),
                "Share": ai.get("share"), "Face readings": ai.get("face_readings"),
                "Staff observation": staff.get("label", "—"), "Review decision": entry["review_decision"],
            })
        st.dataframe(
            pd.DataFrame(rows), use_container_width=True, hide_index=True, height=320,
            column_config={"Share": st.column_config.NumberColumn(
                format="%.2f", help="Aggregated model probability share; not a calibrated confidence or a person count."
            )},
        )
        by_id = {entry["window_id"]: entry for entry in visible}
        playback_key = f"timeline_window_{video_id}"
        if st.session_state.get(playback_key) not in by_id:
            st.session_state[playback_key] = visible[0]["window_id"]

        def timestamp_label(window_id):
            entry = by_id[window_id]
            ai_label = (entry["ai_expression"] or {}).get("label", "No AI suggestion")
            return f"{entry['start_time']} – {entry['end_time']} · {ai_label}"

        selected_id = st.selectbox(
            "Play timestamped window", list(by_id), format_func=timestamp_label, key=playback_key,
        )
        entry = by_id[selected_id]
        st.caption(
            f"Source interval: {entry['start_time']} – {entry['end_time']} "
            f"({entry['start_s']:g}–{entry['end_s']:g} seconds). The clip player starts at 0 for this interval."
        )
        left, right = st.columns([3, 2])
        with left:
            try:
                st.video(_subclip(windows[selected_id], video["source_file"], False))
            except Exception as exc:  # noqa: BLE001
                st.error(f"Could not play this interval: {exc}")
        with right:
            ai = entry["ai_expression"]
            if ai:
                st.warning(
                    f"AI expression: {ai['label']} (unverified). "
                    f"Share {ai.get('share', 0):.2f}; {ai.get('face_readings', 0)} face readings."
                )
                st.caption(f"Model: {ai.get('model_id', 'unknown')}")
            else:
                st.info("No AI suggestion for this interval. Missing evidence is not a Neutral tag.")
            staff = entry["staff_expression"]
            if staff:
                st.write(f"Staff observation: **{staff['label']}**")
                st.caption(f"Recorded by {staff.get('reviewer') or 'unknown'} at {staff.get('reviewed_at') or 'unknown'}.")
            else:
                st.caption("No staff expression observation recorded for this interval.")
            st.caption("Use Review clips → Edit the labels to record a staff observation.")
    else:
        st.info("No timestamped windows match these expression tags. Clear the filter to see all windows.")
    st.caption("Downloads include every window in this video, including intervals without an AI suggestion.")
    st.download_button(
        "Download expression timeline (JSON)", data=json.dumps(document, indent=2),
        file_name=f"{video_id}_expressions.json", mime="application/json",
    )
    st.download_button(
        "Download expression timeline (CSV)", data=expression_timeline_csv(document),
        file_name=f"{video_id}_expressions.csv", mime="text/csv",
    )


def main() -> None:
    st.sidebar.title("greenroom staff review")
    app_run = st.session_state.get("review_app_run", 0) + 1
    st.session_state["review_app_run"] = app_run
    with st.sidebar:
        _upload_panel(app_run)
    reviewer = st.sidebar.text_input("Reviewer name", value=st.session_state.get("reviewer", "")).strip()
    st.session_state["reviewer"] = reviewer

    try:
        index_path = config.windows_index_path()
        if not index_path.is_file():
            st.info("Use Add videos in the sidebar to upload your first video. It will appear here when processing finishes.")
            st.stop()
        index = _index(str(index_path), index_path.stat().st_mtime_ns)
    except Exception as exc:  # noqa: BLE001
        st.error(str(exc))
        st.stop()

    pending_uploads = {job["video_id"] for job in st.session_state.get("upload_jobs", []) if job["status"] != "completed"}
    videos = [v for v in index["videos"] if v["video_id"] not in pending_uploads
              and not (st.session_state.get("upload_queue_error") and v.get("upload_job_id"))
              and (config.records_dir() / v["video_id"] / "records.jsonld").is_file()]
    if not videos:
        st.info("No videos are ready to review yet. Add videos in the sidebar or check Upload progress.")
        st.stop()

    view = st.sidebar.radio("View", ["Review clips", "Expression timeline"], key="review_view")
    if not reviewer and view == "Review clips":
        st.info("Enter your name in the sidebar to review clips, or choose Expression timeline to browse timestamped tags.")
        st.stop()

    def video_label(vid: str) -> str:
        n = next(v["n_windows"] for v in videos if v["video_id"] == vid)
        return f"{vid}  ({len(ledger.latest_decisions(vid))}/{n} reviewed)"

    video_ids = [v["video_id"] for v in videos]
    if st.session_state.get("review_video_id") not in video_ids:
        st.session_state["review_video_id"] = video_ids[0]
    video_id = st.sidebar.selectbox("Video", video_ids, format_func=video_label, key="review_video_id")
    video = next(v for v in videos if v["video_id"] == video_id)
    records = load_records(config.records_dir() / video_id)
    windows = {w["window_id"]: w for w in windows_for_video(index, video_id)}
    decided = ledger.latest_decisions(video_id)

    if view == "Expression timeline":
        _expression_timeline(video_id, video, records, decided, windows)
        return

    queue_name = st.sidebar.radio("Queue", list(QUEUES), index=list(QUEUES).index("Everything"))
    pending_only = st.sidebar.toggle("Pending only", value=True)
    search = st.sidebar.text_input("Filter by label or reason", value="").strip().lower()
    context = st.sidebar.toggle(f"Play with {CONTEXT_PAD_S:g} s of context", value=False)
    show_expression = st.sidebar.toggle(
        "Show unverified AI expression suggestion",
        value=False,
        key="show_expression_suggestions",
        help="Keep expression suggestions visible while browsing windows and videos. These are unverified model outputs.",
    )
    expression_count = sum(bool(r.get("expression_suggestion")) for r in records)
    st.sidebar.caption(f"Expression suggestions available: {expression_count}/{len(records)} windows.")

    statuses = QUEUES[queue_name]
    queue = [r for r in records if statuses is None or r["review_status"] in statuses]
    if pending_only:
        queue = [r for r in queue if r["window_id"] not in decided]
    if search:
        queue = [
            r
            for r in queue
            if search in _name(r["primary_activity"]).lower()
            or any(search in _name(s).lower() for s in r["secondary_activities"])
            or any(search in reason.lower() for reason in r.get("review_reasons", []))
        ]

    chain = ledger.verify_chain(video_id)
    st.sidebar.caption(
        f"Ledger: {chain['entries']} decisions, chain {'intact' if chain['ok'] else 'BROKEN'}. Guideline v{T.GUIDELINE_VERSION}."
    )

    counts = pd.Series([r["review_status"] for r in records]).value_counts()
    st.sidebar.write(
        f"Windows: {len(records)}  |  review required {counts.get('staff_review_required', 0)}  |  "
        f"mixed {counts.get('mixed_activity', 0)}  |  confident {counts.get('ai_suggested', 0)}"
    )

    if not queue:
        st.success("Nothing in this queue. Pick another queue or turn off 'Pending only'.")
        _aggregates(video_id, video, records, decided)
        st.stop()

    pos_key = f"pos::{video_id}::{queue_name}::{pending_only}::{search}"
    positions: dict = st.session_state.setdefault("positions", {})
    if pos_key not in positions:
        positions[pos_key] = _first_pending(queue, decided) or 0
    pos = max(0, min(int(positions[pos_key]), len(queue) - 1))

    nav1, nav2, nav3, nav4 = st.columns([1, 1, 1.4, 2])
    if nav1.button("Previous", use_container_width=True, disabled=pos == 0):
        positions[pos_key] = pos - 1
        st.rerun()
    if nav2.button("Next", use_container_width=True, disabled=pos >= len(queue) - 1):
        positions[pos_key] = pos + 1
        st.rerun()
    if nav3.button("Next pending", use_container_width=True):
        nxt = _first_pending(queue, decided, start_at=pos + 1)
        if nxt is None:
            st.toast("Everything in this queue has a decision.")
        else:
            positions[pos_key] = nxt
            st.rerun()
    jump = nav4.number_input("Item in queue", min_value=1, max_value=len(queue), value=pos + 1, step=1)
    if int(jump) - 1 != pos:
        positions[pos_key] = int(jump) - 1
        st.rerun()

    record = queue[pos]
    window = windows[record["window_id"]]
    previous = decided.get(record["window_id"])
    shown_at: dict = st.session_state.setdefault("shown_at", {})
    shown_at.setdefault(record["window_id"], time.time())

    st.subheader(f"{video_id}  |  {window['start_s']:.0f} s to {window['end_s']:.0f} s  |  item {pos + 1} of {len(queue)}")

    left, right = st.columns([3, 2])
    with left:
        try:
            st.video(_subclip(window, video["source_file"], context))
        except Exception as exc:  # noqa: BLE001
            st.error(f"Could not cut the clip: {exc}")
        if show_expression:
            suggestion = record.get("expression_suggestion")
            if suggestion:
                st.warning(
                    f"{suggestion['label']} (share {suggestion.get('share', 0):.2f}, "
                    f"{suggestion.get('face_readings', 0)} face readings). Unverified model output, "
                    "not a statement about how anyone feels. Use the Edit form to record a staff observation."
                )
            else:
                st.info(
                    "No expression suggestion is available for this window. "
                    "Tagging must have run and found enough usable face readings to produce a suggestion."
                )
        st.markdown(f"**{STATUS_TEXT.get(record['review_status'], record['review_status'])}**")
        st.markdown(
            f"Suggested primary: **{_name(record['primary_activity'])}**  (P = {record['confidence']:.2f}, "
            f"belief {record['opinion_primary']['b']:.2f}, disbelief {record['opinion_primary']['d']:.2f}, "
            f"uncertainty {record['opinion_primary']['u']:.2f}, conflict {record['conflict_score']:.2f})"
        )
        if record["secondary_activities"]:
            st.markdown("Also suggested: " + ", ".join(_name(s) for s in record["secondary_activities"]))
        if record.get("review_reasons"):
            st.caption("Why this is in the queue: " + "; ".join(record["review_reasons"]))
        speech = record.get("speech")
        if speech:
            flag = record.get("facilitation_flag")
            st.caption(
                f"Speech: {speech.get('speech_kind', 'unclear')} ({speech.get('word_count', 0)} words); "
                f"facilitation: {'yes' if flag else 'no' if flag is False else 'unclear'} "
                f"(confidence {speech.get('facilitation_confidence', 0.0):.2f}). {speech.get('rationale', '')}"
            )
        with st.expander("Evidence behind the suggestion"):
            st.caption("Fused opinions, top labels")
            st.dataframe(_label_table(record), use_container_width=True, hide_index=True)
            st.caption(f"Per channel, for {_name(record['primary_activity'])} (uncertainty 1.0 means the channel did not score it)")
            st.dataframe(_channel_table(record, record["primary_activity"]), use_container_width=True, hide_index=True)

    with right:
        if previous:
            st.info(
                f"Decision on file: {previous['decision']} by {previous['reviewer']} at {previous['timestamp']}"
                + (f"; final label {_name(previous['final_primary'])}" if previous.get("final_primary") else "")
            )
        note = st.text_input("Reviewer note (optional)", key=f"note_{record['window_id']}")
        b1, b2, b3 = st.columns(3)
        decision = None
        if b1.button("Accept", type="primary", use_container_width=True):
            decision = ("accepted", record["primary_activity"], record["secondary_activities"], None, None, None)
        if b2.button("Reject", use_container_width=True):
            decision = ("rejected", None, [], None, None, None)
        if b3.button("Defer", use_container_width=True):
            decision = ("deferred", None, [], None, None, None)

        with st.expander("Edit the labels", expanded=False):
            with st.form(key=f"edit_{record['window_id']}"):
                default_primary = record["primary_activity"]
                primary = st.radio(
                    "Primary",
                    T.ALL_LABEL_IDS,
                    index=T.ALL_LABEL_IDS.index(default_primary) if default_primary in T.ALL_LABEL_IDS else 0,
                    format_func=_name,
                )
                secondary = st.multiselect(
                    "Secondary activities",
                    T.ACTIVITY_IDS,
                    default=[s for s in record["secondary_activities"] if s in T.ACTIVITY_IDS],
                    format_func=_name,
                )
                facilitation = st.checkbox("Facilitation present (instructor or peer directing, teaching, giving feedback)")
                observation = st.text_input("Staff contextual observation (optional, staff-only)")
                expression = st.selectbox(
                    "Staff-only contextual expression (optional; never an automated claim)",
                    ["none"] + T.EXPRESSION_LABELS,
                )
                edit_note = st.text_input("Note for this edit (optional)")
                submitted = st.form_submit_button("Save edit", use_container_width=True)
            if submitted:
                problems = T.validate_label_set(primary, secondary)
                if problems:
                    st.error("; ".join(problems))
                else:
                    decision = (
                        "edited",
                        primary,
                        secondary,
                        facilitation,
                        observation,
                        None if expression == "none" else expression,
                    )
                    note = edit_note or note

        if decision is not None:
            kind, final_primary, final_secondary, facilitation, observation, expression = decision
            ledger.append_decision(
                video_id,
                window,
                reviewer,
                kind,
                record,
                final_primary,
                list(final_secondary),
                facilitation,
                observation,
                expression,
                note or "",
                time.time() - shown_at.get(record["window_id"], time.time()),
            )
            shown_at.pop(record["window_id"], None)
            decided = ledger.latest_decisions(video_id)
            if pending_only:
                # the decided item leaves the queue on rerun, so the next item slides into this position
                positions[pos_key] = pos
            else:
                nxt = _first_pending(queue, decided, start_at=pos + 1)
                positions[pos_key] = nxt if nxt is not None else min(pos + 1, len(queue) - 1)
            st.toast(f"Recorded: {kind}")
            st.rerun()

    _aggregates(video_id, video, records, decided)


def _aggregates(video_id: str, video: dict, records: list[dict], decided: dict) -> None:
    with st.expander("Session summary and export", expanded=False):
        win_s = video["duration_s"] / max(1, video["n_windows"])
        by_label = pd.Series([r["primary_activity"] for r in records]).value_counts()
        summary = pd.DataFrame(
            {
                "label": [_name(l) for l in by_label.index],
                "windows (AI primary)": by_label.values,
                "minutes": [round(c * win_s / 60.0, 1) for c in by_label.values],
            }
        )
        reviewed = [decided[r["window_id"]] for r in records if r["window_id"] in decided]
        if reviewed:
            dec = pd.Series([d["decision"] for d in reviewed]).value_counts()
            agree = sum(
                1 for d in reviewed if d["decision"] == "accepted" or (d["decision"] == "edited" and d["final_primary"] == d["ai_primary"])
            )
            judged = sum(1 for d in reviewed if d["decision"] in ("accepted", "edited", "rejected"))
            st.write(
                f"Decisions: {dict(dec)}.  Primary label agreed by staff in {agree} of {judged} judged windows"
                + (f" ({100.0 * agree / judged:.0f}%)." if judged else ".")
            )
            spent = [d["seconds_spent"] for d in reviewed if d.get("seconds_spent") is not None]
            if spent:
                st.write(f"Median seconds per decision: {pd.Series(spent).median():.1f} (n = {len(spent)}).")
        st.dataframe(summary, use_container_width=True, hide_index=True)
        st.caption("Session-level counts only. No per-participant statistics exist anywhere in this system.")
        merged = _merged_export(records, decided)
        st.download_button(
            "Download records with decisions (JSON)",
            data=json.dumps(merged, indent=2),
            file_name=f"{video_id}_reviewed.json",
            mime="application/json",
        )
        st.download_button(
            "Download records with decisions (CSV)",
            data=pd.DataFrame(merged).to_csv(index=False),
            file_name=f"{video_id}_reviewed.csv",
            mime="text/csv",
        )


main()
