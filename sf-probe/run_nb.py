"""Headless notebook run: python run_nb.py probe_mlp.ipynb"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / ".pylibs"))
import nbformat
from nbclient import NotebookClient
p = Path(sys.argv[1])
nb = nbformat.read(p, as_version=4)
NotebookClient(nb, timeout=8 * 3600, kernel_name="python3",
               resources={"metadata": {"path": str(p.parent)}}).execute()
nbformat.write(nb, p)
print("notebook done")
