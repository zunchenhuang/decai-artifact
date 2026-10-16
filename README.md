# Artifact: dec-AI

## 1. Setup (macOS or Linux, CPU only)

The models are small (85k–568k parameters), so neither a GPU nor Apple MPS is needed.

```bash
cd artifact
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

With conda, one Python 3.11 environment runs everything, including the auto_LiRPA stages:

```bash
conda create -y -n sst311 python=3.11 && conda activate sst311
pip install -r requirements.txt "transformers>=4.51,<5" safetensors sympy packaging tqdm appdirs graphviz
pip install --no-deps "git+https://github.com/Verified-Intelligence/auto_LiRPA.git@5a098e8"
python -c "import auto_LiRPA, numpy, torch; print(auto_LiRPA.__version__, numpy.__version__, torch.__version__)"
THREADS=8 PYLIRPA="$(which python)" ./run_all.sh --skip-training
```

The core (everything except auto_LiRPA) also runs on Python 3.12 with numpy 1.x or 2.x.

## 2. Quick check (about 2 minutes, uses the bundled models)

```bash
make check
```

- `tests/check_tibp.py`: the knapsack bound never undercuts brute force, and the bounds are exact on
  single inputs and contain every sampled concrete input (every line: `max violation 0.00e+00`).
- `src/example_concrete.py`: the running example (Sect. 3).
- `gpt2/test_equiv.py`, `gpt2/test_soundness.py` (if transformers is installed): `ALL OK` and
  `TOTAL violations: 0 | refinement agrees with ground truth: 42 / 42`.
- `lean/SoundDecoding.lean` (if lean is installed): `Lean: OK`.

## 3. Full reproduction

```bash
THREADS=8 ./run_all.sh --skip-training                              # evaluate the bundled models (recommended)
THREADS=8 PYLIRPA=~/venv311/bin/python ./run_all.sh --skip-training   # ... including the auto_LiRPA stages
THREADS=8 ./run_all.sh                                              # retrain every model, then evaluate
```
