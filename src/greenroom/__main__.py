"""Allow `python -m greenroom ...`."""
import sys

from .cli import main

sys.exit(main())
