"""Stable launcher for the Interpretability dashboard.

This keeps Flask's development reloader disabled so the persistent training
queue and its worker are initialized exactly once.
"""

from __future__ import annotations

import socket


HOST = "127.0.0.1"
PORT = 5000


def port_is_open(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as connection:
        connection.settimeout(0.25)
        return connection.connect_ex((host, port)) == 0


if __name__ == "__main__":
    if port_is_open(HOST, PORT):
        print(f"Dashboard is already running at http://{HOST}:{PORT}/")
    else:
        from server import app

        print(f"Dashboard: http://{HOST}:{PORT}/", flush=True)
        app.run(
            host=HOST,
            port=PORT,
            debug=False,
            use_reloader=False,
            threaded=True,
        )
