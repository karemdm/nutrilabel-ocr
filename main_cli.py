"""Run nutrilabel from the terminal without installing it (no GUI).

    python main_cli.py read path/to/label.jpg
    python main_cli.py read path/to/folder --csv results.csv --out-dir annotated/
    python main_cli.py engines

Equivalent to the ``nutrilabel`` command once the package is installed.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from nutrilabel.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
