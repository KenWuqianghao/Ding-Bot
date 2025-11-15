# Installing Ding-Bot Dependencies

## Using uv (Fast & Recommended)

```bash
cd /home/kenwu/Chess-Bot/Ding-Bot
uv pip install -r requirements.txt
```

This installs all dependencies into the parent `.venv` environment.

## Using pip (Alternative)

```bash
cd /home/kenwu/Chess-Bot/Ding-Bot
source ../.venv/bin/activate
pip install -r requirements.txt
```

## Dependencies Installed

- `fastapi` - Web framework for the API
- `uvicorn` - ASGI server
- `python-chess` - Chess library
- `pydantic` - Data validation
- And all other dependencies from `requirements.txt`

## Note

Ding-Bot uses the parent Chess-Bot's virtual environment (`.venv`), so all dependencies are installed there. This allows Ding-Bot to also access Chess-Bot's dependencies like `torch` for the neural network.

## Verify Installation

```bash
cd /home/kenwu/Chess-Bot
source .venv/bin/activate
python -c "import fastapi; print('FastAPI:', fastapi.__version__)"
python -c "import uvicorn; print('Uvicorn:', uvicorn.__version__)"
```

## Running Ding-Bot

After installing dependencies:

```bash
cd /home/kenwu/Chess-Bot/Ding-Bot/devtools
npm run dev
```

The backend (`serve.py`) should now start successfully!

