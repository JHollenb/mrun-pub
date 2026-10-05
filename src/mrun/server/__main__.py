"""``python -m mrun.server`` — run the scheduler with uvicorn."""

import os

import uvicorn


def main() -> None:
    uvicorn.run(
        "mrun.server.app:app",
        host=os.environ.get("MRUN_BIND", "127.0.0.1"),
        port=int(os.environ.get("MRUN_PORT", "9025")),
        log_level="info",
    )


if __name__ == "__main__":
    main()
