"""One place that knows the layout. Everything else asks here.

    PLL_Attempt/
      src/                  this file and every module
      data/                 *.npz
      Hyperparameter_sweep/ sweeps_*/
      config/  runs/  graphs/  hpc/  docs/  PINNs-in-EMT/

`data()` and `sweeps()` resolve a BARE name into its folder and leave an explicit
path alone. That is what keeps `--dataset famB_W40.npz --results_dir sweeps_famB_ff`
working unchanged, so no hpc config file, and no `ckpt` field in an existing JSON
record, had to be rewritten for the move.

`runs/` deliberately stays at the project root: every sweep record stores
`"ckpt": "runs/<tag>.pth"` as a root-relative path, and moving it would invalidate all
of them. Run scripts from the project root.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONFIG = ROOT / "config"
DATA = ROOT / "data"
SWEEPS = ROOT / "Hyperparameter_sweep"
RUNS = ROOT / "runs"
GRAPHS = ROOT / "graphs"
PAPER_REPO = ROOT / "PINNs-in-EMT"


def _under(root, p):
    p = Path(p)
    return root / p if p.parent == Path(".") else p


def data(p):
    """'famB_W40.npz' -> data/famB_W40.npz ; 'some/dir/x.npz' -> unchanged."""
    return _under(DATA, p)


def sweeps(p):
    """'sweeps_famB_ff' -> Hyperparameter_sweep/sweeps_famB_ff ; a path -> unchanged."""
    return _under(SWEEPS, p)


def graphs(p):
    """'graphs/09_x.png' or '09_x.png' -> graphs/09_x.png at the project root."""
    p = Path(p)
    return GRAPHS / (p.name if p.parent in (Path("."), Path("graphs")) else p)
