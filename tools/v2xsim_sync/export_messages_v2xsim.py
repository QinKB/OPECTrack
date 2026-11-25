"""Export V2X-Sim roadside messages and training supervision."""
import sys
from pathlib import Path

import torch

PAIRS = Path(sys.argv.pop(1))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "opectrack"))
import export_messages as em  # noqa: E402

em.spd_pairs.cached = lambda root, cache, split_file=None, split=None: torch.load(PAIRS / f"pairs_{split}.pt")
em.main()
