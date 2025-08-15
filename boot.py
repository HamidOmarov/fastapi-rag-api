import os, sys, importlib
import uvicorn

candidates = [
    os.getenv("APP_MODULE"),
    "app.main:app",
    "main:app",
    "app.api:app",
    "app.server:app",
]

def try_run(spec: str):
    mod, name = spec.split(":")
    m = importlib.import_module(mod)
    app = getattr(m, name)
    port = int(os.getenv("PORT", "7860"))
    uvicorn.run(app, host="0.0.0.0", port=port)

if __name__ == "__main__":
    last_err = None
    for cand in [c for c in candidates if c]:
        try:
            print(f"[boot] Trying {cand}", flush=True)
            try_run(cand)
            sys.exit(0)
        except Exception as e:
            print(f"[boot] Failed {cand}: {e}", flush=True)
            last_err = e
    print("[boot] No valid APP_MODULE found. Set env APP_MODULE='pkg.module:app'", flush=True)
    if last_err:
        raise last_err
    sys.exit(1)

