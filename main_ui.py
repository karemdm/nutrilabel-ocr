"""Open the local web interface without installing the package.

    python main_ui.py                     # http://127.0.0.1:8765, opens your browser
    python main_ui.py --engine paddleocr --port 9000

Equivalent to ``nutrilabel ui`` once the package is installed.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from nutrilabel.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main(["ui", *sys.argv[1:]]))
