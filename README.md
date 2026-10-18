# Gradient-free hyperdimensional session adaptation on FALCON H1, M1, M2

Code and result records for the study of closed-form session adaptation of a frozen causal recurrent decoder on the FALCON tasks H1, M1 and M2. Methods compared on a chronological stream of held-out sessions: `hdc` (proposed), `closed_linear`, `hdc_bound`, `frozen`, `finetune_full`, `finetune_readout`, `ewc`, `replay`, and the FALCON reference decoders WF and RNN.

## Hardware

NVIDIA RTX 4080 Laptop GPU, 12 GB. Python 3.10, torch 2.5.1 (CUDA 12.4). Pinned packages are in `environment/requirements.lock`.

## Environment

    python3.10 -m venv ../.venv
    ../.venv/bin/python -m pip install -r environment/requirements.lock
    ../.venv/bin/python -m pip install falcon-challenge==1.0.2
    mkdir -p ../falcon-challenge/data

The loader is the `falcon_challenge` package, version 1.0.2 from PyPI; data live in `../falcon-challenge/data`. The manuscript steps (tikz, strip, docx, audit, manifest) need the private `paper/` sources and system tools (`pandoc`, LibreOffice `soffice`, poppler `pdftoppm`, `qpdf`, PyYAML, lxml, Pillow); they are not needed to reproduce the results.

## Data

NWB files are downloaded from DANDI into `../falcon-challenge/data/<dandiset>`: H1 000954, M1 000941, M2 000953. File hashes are recorded in `results/provenance/manifest_{task}.json`. The data are not redistributed here.

## Reproduce

    make analysis

`make analysis` runs every stage that produces results; `make all` also renders the manuscript. Stages run in order and skip result files that already exist. Set `TASKS="h1"` to restrict tasks and `PY=` to choose the interpreter.

| Target | Script | Produces | Runtime |
| --- | --- | --- | --- |
| data | dandi CLI | NWB files | network bound |
| cache | `code/data.py` | `cache/*.npz`, provenance manifests | 5 min |
| overlap | `code/minival_overlap.py` | `results/provenance/minival_overlap.json` | 1 min |
| select | `code/run.py select` | `results/selection/{task}.json` (nested selection on held-in sessions only) | 80 min |
| main | `code/run.py main` | `results/stream/{task}/full/*.json`, 10 seeds | 75 min |
| ablation | `code/run.py ablation` | `results/stream/{task}/{variant}/*.json`, 5 seeds | 75 min |
| budget | `code/run.py budget` | `results/stream/{task}/epochs{25,100}/*.json` | 40 min |
| dose | `code/run.py dose` | `results/dose/{task}/*.json` | 90 min |
| baselines | `code/falcon_baselines.py` | `results/stream/{task}/falcon/wf_*.json`; RNN references from the commands in `code/README_runs.txt` | 20 min |
| stats | `code/stats.py` | `results/stats/*.json` | 2 min |
| tables | `code/tables.py` | `results/tables/*.md` | seconds |
| figures | `code/figures.py` | `paper/figures/f*.png`, `f*.pdf` | 1 min |
| tikz | `paper/tikz/build.sh` | `paper/tikz/d*.png`, copied to `paper/figures/` | 1 min |
| theory | `code/theory_check.py` | `results/theory/bound.json` | 5 min |
| macs | `code/verify_macs.py` | `results/macs/verify.json`, `results/macs/verify.md` | 1 min |
| docx | `code/assemble.py`, `code/build_docx.py` | `paper/manuscript.docx` | seconds |
| audit | `code/audit_docx.py` | template and consistency report | seconds |
| manifest | `code/manifest.py` | `MANIFEST.sha256` | seconds |

Runtimes are for the hardware above. Manuscript sources (`paper/`) are not part of this repository; the figures, tikz, docx and audit stages write into `paper/`. Pretrained backbone checkpoints (`results/checkpoints`) and the EvalAI decoder artifacts (`submission/artifacts`) are in the v1.0 release archive (https://github.com/nGQD/falcon-hdc/releases/tag/v1.0); extract it in the repository root.

## License

AGPL-3.0, see `LICENSE`.
