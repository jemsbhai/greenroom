"""Gold-mode labeling app: human labels on fixed windows, with no model output on screen.

Run (from the repo root, inside the venv):
    python -m greenroom windows --subclips
    python -m streamlit run app/gold_app.py --server.address 127.0.0.1

Labels are appended to <data>/gold/<annotator>.jsonl; the latest record per
window wins. The app never shows AI suggestions, so the resulting labels can
serve as the held-out evaluation set.
"""
from __future__ import annotations

import time

import pandas as pd
import streamlit as st

from greenroom import config, gold
from greenroom import taxonomy as T
from greenroom.media import cut_subclip
from greenroom.windows import load_index, windows_for_video

st.set_page_config(page_title="greenroom | gold labeling", layout="wide")

CONTEXT_PAD_S = 5.0


@st.cache_data(show_spinner=False)
def _load_index(path: str) -> dict:
    return load_index(path)


def _subclip_for(window: dict, source_file: str, context: bool) -> str:
    video_path = config.videos_dir() / source_file
    folder = config.subclips_dir() / window["video_id"]
    if context:
        start = max(0.0, window["start_s"] - CONTEXT_PAD_S)
        duration = (window["end_s"] - window["start_s"]) + 2 * CONTEXT_PAD_S
        dst = folder / f"ctx_{window['index']:05d}.mp4"
    else:
        start = window["start_s"]
        duration = window["end_s"] - window["start_s"]
        dst = folder / f"{window['index']:05d}.mp4"
    return str(cut_subclip(video_path, dst, start, duration))


def _first_unlabeled(windows: list[dict], latest: dict[str, dict], start_at: int = 0) -> int | None:
    for i in range(start_at, len(windows)):
        if windows[i]["window_id"] not in latest:
            return i
    for i in range(0, min(start_at, len(windows))):
        if windows[i]["window_id"] not in latest:
            return i
    return None


def main() -> None:
    try:
        index = _load_index(str(config.windows_index_path()))
    except Exception as exc:  # noqa: BLE001
        st.error(f"{exc}")
        st.code("python -m greenroom windows --subclips")
        st.stop()

    st.sidebar.title("greenroom gold labeling")
    annotator = st.sidebar.text_input("Your name (annotator)", value=st.session_state.get("annotator", "")).strip()
    st.session_state["annotator"] = annotator
    if not annotator:
        st.info("Enter your name in the sidebar to start. Labels are saved per annotator.")
        st.stop()

    latest = gold.load_latest(annotator)
    video_rows = index["videos"]
    labels_for = {
        v["video_id"]: f"{v['video_id']}  ({gold.progress(latest, windows_for_video(index, v['video_id']))[0]}/{v['n_windows']})"
        for v in video_rows
    }
    video_id = st.sidebar.selectbox(
        "Video", [v["video_id"] for v in video_rows], format_func=lambda vid: labels_for[vid]
    )
    source_file = next(v["source_file"] for v in video_rows if v["video_id"] == video_id)
    windows = windows_for_video(index, video_id)
    if not windows:
        st.warning("This video has no windows.")
        st.stop()

    context = st.sidebar.toggle(f"Play with {CONTEXT_PAD_S:g} s of context before and after", value=False)
    st.sidebar.caption(
        "Label what is visible or audible inside the window only. Context is for understanding, not for labeling."
    )

    # Position within this video
    positions: dict = st.session_state.setdefault("positions", {})
    if video_id not in positions:
        positions[video_id] = _first_unlabeled(windows, latest) or 0
    pos = int(positions[video_id])
    pos = max(0, min(pos, len(windows) - 1))

    nav1, nav2, nav3, nav4 = st.columns([1, 1, 1.4, 2])
    if nav1.button("Previous", use_container_width=True, disabled=pos == 0):
        positions[video_id] = pos - 1
        st.rerun()
    if nav2.button("Next", use_container_width=True, disabled=pos >= len(windows) - 1):
        positions[video_id] = pos + 1
        st.rerun()
    if nav3.button("Next unlabeled", use_container_width=True):
        nxt = _first_unlabeled(windows, latest, start_at=pos + 1)
        if nxt is None:
            st.toast("Every window in this video is labeled.")
        else:
            positions[video_id] = nxt
            st.rerun()
    jump = nav4.number_input("Jump to window", min_value=1, max_value=len(windows), value=pos + 1, step=1)
    if int(jump) - 1 != pos:
        positions[video_id] = int(jump) - 1
        st.rerun()

    window = windows[pos]
    done, total = gold.progress(latest, windows)
    st.subheader(
        f"{video_id}  |  window {pos + 1} of {total}  |  {window['start_s']:.1f} s to {window['end_s']:.1f} s"
    )
    st.progress(done / total if total else 0.0, text=f"{done} of {total} windows labeled by {annotator}")

    shown_at: dict = st.session_state.setdefault("shown_at", {})
    shown_at.setdefault(window["window_id"], time.time())

    left, right = st.columns([3, 2])
    with left:
        try:
            st.video(_subclip_for(window, source_file, context))
        except Exception as exc:  # noqa: BLE001
            st.error(f"Could not cut the subclip: {exc}")
        if context:
            st.caption(
                f"Context clip: the window is the middle {window['end_s'] - window['start_s']:.0f} s "
                f"(from {CONTEXT_PAD_S:g} s to {CONTEXT_PAD_S + window['end_s'] - window['start_s']:g} s of the clip)."
            )

    previous = latest.get(window["window_id"])
    with right:
        with st.form(key=f"form_{window['window_id']}"):
            default_primary = previous["primary"] if previous else T.PRIMARY_CHOICES[-1][0]
            primary = st.radio(
                "Primary (dominant content of the window)",
                T.ALL_LABEL_IDS,
                index=T.ALL_LABEL_IDS.index(default_primary) if default_primary in T.ALL_LABEL_IDS else 0,
                format_func=lambda lid: T.DISPLAY_NAME[lid],
            )
            secondary = st.multiselect(
                "Secondary activities (also present)",
                T.ACTIVITY_IDS,
                default=[s for s in (previous["secondary"] if previous else []) if s in T.ACTIVITY_IDS],
                format_func=lambda lid: T.DISPLAY_NAME[lid],
            )
            facilitation = st.checkbox(
                "Facilitation present (an instructor or peer directing, teaching or giving feedback)",
                value=bool(previous["facilitation"]) if previous else False,
            )
            uncertain = st.checkbox(
                "Uncertain (I cannot tell from this window, even with context)",
                value=bool(previous["uncertain"]) if previous else False,
            )
            note = st.text_input(
                "Note (optional; no names, no descriptions that identify a person)",
                value=previous["note"] if previous else "",
            )
            submitted = st.form_submit_button("Save and go to next unlabeled", type="primary", use_container_width=True)

        if submitted:
            problems = T.validate_label_set(primary, secondary)
            if problems:
                st.error("; ".join(problems))
            else:
                record = gold.make_record(
                    window,
                    annotator,
                    primary,
                    secondary,
                    facilitation,
                    uncertain,
                    note=note,
                    context_used=context,
                    seconds_spent=time.time() - shown_at.get(window["window_id"], time.time()),
                )
                gold.append_record(record)
                latest[window["window_id"]] = record
                shown_at.pop(window["window_id"], None)
                nxt = _first_unlabeled(windows, latest, start_at=pos + 1)
                positions[video_id] = nxt if nxt is not None else min(pos + 1, len(windows) - 1)
                st.toast("Saved.")
                st.rerun()

        if previous:
            st.caption(f"Previously saved {previous['timestamp']} (editing appends a new record; history is kept).")

    with st.expander("Your labels for this video"):
        rows = [
            {
                "window": w["index"] + 1,
                "start_s": w["start_s"],
                "primary": T.DISPLAY_NAME.get(latest[w["window_id"]]["primary"], latest[w["window_id"]]["primary"]),
                "secondary": ", ".join(latest[w["window_id"]]["secondary"]),
                "facilitation": latest[w["window_id"]]["facilitation"],
                "uncertain": latest[w["window_id"]]["uncertain"],
            }
            for w in windows
            if w["window_id"] in latest
        ]
        if rows:
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        else:
            st.write("Nothing saved yet for this video.")

    with st.sidebar.expander("Label counts so far (all videos)"):
        counts = pd.Series([r["primary"] for r in latest.values()]).value_counts() if latest else pd.Series(dtype=int)
        if len(counts):
            st.dataframe(counts.rename("windows"), use_container_width=True)
        else:
            st.write("No labels yet.")
    st.sidebar.caption(f"Guideline version {T.GUIDELINE_VERSION}. Gold files: {config.gold_dir()}")


main()
