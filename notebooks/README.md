# notebooks/

Notebooks are for exploring and for presenting. Logic that more than one notebook
needs, or that a script needs, goes into `src/zfpv/` and is imported.

## Conventions

- **Name:** `NN-topic-initials.ipynb`, e.g. `01-audit-ak.ipynb`, `02-baseline-yolo-ol.ipynb`.
- **One owner per notebook.** Notebook merge conflicts are unreadable JSON. If you want
  to change your teammate's notebook, ask, or copy the idea into your own.
- **Outputs are stripped on commit** (`nbstripout` in `.pre-commit-config.yaml`). Any
  plot or table that matters gets saved to `results/figures/` or `results/` from the
  notebook, and that file is committed. (The course template's default is to commit
  outputs; we turned that off to avoid merge noise. It's a team decision, revisit it if
  reviewers need to see plots on GitHub.)
- **Top cell** always sets the working directory to the repo root and loads the config:

  ```python
  from zfpv.config import load_config
  from zfpv.utils import set_seed
  cfg = load_config()
  set_seed(cfg["seed"])
  ```

## Running on Colab

First cell of any notebook that trains on Colab:

```python
!git clone https://github.com/Luzefik/storozha-anti-ambush.git
%cd storozha-anti-ambush
!git checkout <your-branch>
!pip install -q -r requirements.txt -r requirements-train.txt -e .

from google.colab import userdata
import os
os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")  # set once under Colab "Secrets"

!python scripts/download_data.py
!python scripts/prepare_data.py   # frames on the runtime's local disk, not on Drive
```

- Save weights and large artifacts to your **private** Google Drive; do not share the
  folder (dataset licence).
- Bring back only small files (`results/metrics/*.json`) and **commit from your
  laptop**, not from Colab.
