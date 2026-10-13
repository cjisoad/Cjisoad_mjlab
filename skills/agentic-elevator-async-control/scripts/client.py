"""Invoke the repository's maintained async client from a skill installation."""
from pathlib import Path
import os
import runpy

configured_root = os.environ.get('UNITREE_AGENTIC_REPO')
if configured_root:
    root = Path(configured_root).expanduser().resolve()
else:
    candidates = [*Path(__file__).resolve().parents, Path.cwd(), *Path.cwd().parents]
    root = next((candidate for candidate in candidates
                 if (candidate / 'scripts' / 'agentic_pipeline_client.py').is_file()), None)
    if root is None:
        raise SystemExit('Cannot find the checkout; set UNITREE_AGENTIC_REPO to its root.')
runpy.run_path(str(root / 'scripts' / 'agentic_pipeline_client.py'), run_name='__main__')
