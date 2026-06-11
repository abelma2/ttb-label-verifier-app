# Marks api/ as a package so local dev can run `uvicorn api.index:app`.
# Vercel only builds non-underscore .py files in api/ as functions, so this file
# (and _models.py) are bundled as plain modules, not endpoints.
