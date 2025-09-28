# MGRS

Utility scripts for reading drone imagery, parsing coordinates, and guiding an
operator through a SALUTE report.

## Development

Create a virtual environment and install dependencies from `requirements.txt`.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Tests

Unit tests live under `tests/` and are executed with `pytest`.

```bash
pytest --maxfail=1 --disable-warnings -q
```
