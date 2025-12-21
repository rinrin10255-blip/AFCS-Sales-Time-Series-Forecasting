from __future__ import annotations

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from forecasting.forecasting_afcs2025 import main


if __name__ == "__main__":
    main()
