"""Run progress beside the run's TensorBoard application."""

import argparse
import json
import pathlib

import werkzeug.wrappers


class Dashboard:
    def __init__(self, root, charts):
        self.root = pathlib.Path(root)
        self.charts = charts

    def __call__(self, environ, start_response):
        path = environ["PATH_INFO"]
        if path == "/tensorboard/":
            response = werkzeug.wrappers.Response(
                pathlib.Path(__file__).with_suffix(".html").read_text(),
                content_type="text/html; charset=utf-8",
            )
        elif path == "/tensorboard/chart-colors.js":
            response = werkzeug.wrappers.Response(
                pathlib.Path(__file__).with_name("chart-colors.js").read_text(),
                content_type="text/javascript; charset=utf-8",
            )
        elif path == "/tensorboard/charts/":
            response = werkzeug.wrappers.Response.from_app(self.charts, environ)
            response.set_data(
                response.get_data().replace(
                    b"</body>",
                    b'<script src="/tensorboard/chart-colors.js"></script></body>',
                )
            )
        elif path == "/tensorboard/status.json":
            try:
                data = (self.root / "monitor_status.json").read_bytes()
            except FileNotFoundError:
                response = werkzeug.wrappers.Response(
                    '{"error":"Monitor has not written status yet."}',
                    status=503,
                    content_type="application/json",
                )
            else:
                response = werkzeug.wrappers.Response(
                    json.dumps({**json.loads(data), "run": self.root.name}),
                    content_type="application/json",
                )
        else:
            return self.charts(environ, start_response)
        response.headers["Cache-Control"] = "no-store"
        return response(environ, start_response)


def run():
    import tensorboard.program

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=pathlib.Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8794)
    args = parser.parse_args()
    tensorboard_app = tensorboard.program.TensorBoard(
        server_class=lambda charts, flags: tensorboard.program.WerkzeugServer(
            Dashboard(args.root, charts), flags
        )
    )
    tensorboard_app.configure(
        argv=[
            "tensorboard",
            "--logdir",
            str(args.root / "tensorboard"),
            "--path_prefix",
            "/tensorboard/charts",
            "--host",
            args.host,
            "--port",
            str(args.port),
        ]
    )
    tensorboard_app.main()


if __name__ == "__main__":
    run()
