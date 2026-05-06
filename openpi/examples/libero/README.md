# LIBERO Benchmark

This example runs the LIBERO benchmark: https://github.com/Lifelong-Robot-Learning/LIBERO

Note: When updating requirements.txt in this directory, there is an additional flag `--extra-index-url https://download.pytorch.org/whl/cu113` that must be added to the `uv pip compile` command.

This example requires git submodules to be initialized. Don't forget to run:

```bash
git submodule update --init --recursive
```

## With Docker (recommended)

```bash
# Grant access to the X11 server:
sudo xhost +local:docker

# To run with the default checkpoint and task suite:
SERVER_ARGS="--env LIBERO" docker compose -f examples/libero/compose.yml up --build

# To run with glx for Mujoco instead (use this if you have egl errors):
MUJOCO_GL=glx SERVER_ARGS="--env LIBERO" docker compose -f examples/libero/compose.yml up --build
```

You can customize the loaded checkpoint by providing additional `SERVER_ARGS` (see `scripts/serve_policy.py`), and the LIBERO client by providing additional `CLIENT_ARGS` (see `examples/libero/main.py`, `Args` dataclass). Run `python examples/libero/main.py --help` for all flags.

Notable client options include `--filename` (path for evaluation logs written by `logging`; default `logs/grid.txt`; parent directories are created automatically), plus connection settings such as `--host` and `--port`.

For example:

```bash
# To load a custom checkpoint (located in the top-level openpi/ directory):
export SERVER_ARGS="--env LIBERO policy:checkpoint --policy.config pi05_libero --policy.dir ./my_custom_checkpoint"

# Custom evaluation log file (combine with --host / --port as needed):
export CLIENT_ARGS="--filename logs/my_libero_eval.txt"
```

## Without Docker (not recommended)

Terminal window 1:

```bash
# Create virtual environment
uv venv --python 3.8 examples/libero/.venv
source examples/libero/.venv/bin/activate
uv pip sync examples/libero/requirements.txt third_party/libero/requirements.txt --extra-index-url https://download.pytorch.org/whl/cu113 --index-strategy=unsafe-best-match
uv pip install -e packages/openpi-client
uv pip install -e third_party/libero
export PYTHONPATH=$PYTHONPATH:$PWD/third_party/libero

# Run the simulation
python examples/libero/main.py --filename logs/my_run.txt

# To run with glx for Mujoco instead (use this if you have egl errors):
MUJOCO_GL=glx python examples/libero/main.py
```

Terminal window 2:

```bash
# Run the server
uv run scripts/serve_policy.py --env LIBERO
```

## Results

| Model | Libero Spatial | Libero Object | Libero Goal | Libero 10 | Average |
|-------|---------------|---------------|-------------|-----------|---------|
| π0.5 @ 30k (Official Finetuned) | 98.8 | 98.2 | 98.0 | 92.4 | 96.9
| π0.5 @ 30k (Our Finetuned) | 98.4 | 98.0 | 97.6 | 92.8 | 96.7
| π0.5 @ 30k (GridS16) | 98.6 | 98.8 | 98.4 | 95.2 | 97.7
| π0 @ 30k (Our Finetuned) | 97.2 | 98.8 | 96.0 | 85.6 | 94.4
| π0 @ 30k (GridS16) | 98.0 | 99.2 | 96.4 | 90.2 | 96.0

