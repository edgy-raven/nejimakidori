"""Run progress beside the run's TensorBoard application."""

import argparse
import json
import pathlib

import werkzeug.wrappers


class Dashboard:
    def __init__(self, root, charts, joint=False):
        self.root = pathlib.Path(root)
        self.charts = charts
        self.joint = joint

    def __call__(self, environ, start_response):
        path = environ["PATH_INFO"]
        static_files = {
            "/tensorboard/": (
                "joint_dashboard.html" if self.joint else "dashboard.html",
                "text/html",
            ),
            "/tensorboard/joint-dashboard.js": (
                "joint-dashboard.js",
                "text/javascript",
            ),
            "/tensorboard/chart-colors.js": (
                "chart-colors.js",
                "text/javascript",
            ),
        }
        if path in static_files:
            filename, content_type = static_files[path]
            response = werkzeug.wrappers.Response(
                pathlib.Path(__file__).with_name(filename).read_text(),
                content_type=content_type + "; charset=utf-8",
            )
        elif path == "/tensorboard/charts/":
            response = werkzeug.wrappers.Response.from_app(self.charts, environ)
            response.set_data(
                response.get_data().replace(
                    b"</body>",
                    b'<script src="/tensorboard/chart-colors.js">'
                    b"</script></body>",
                )
            )
        elif path == "/tensorboard/status.json" and self.joint:
            response = werkzeug.wrappers.Response(
                json.dumps(
                    {
                        "run": self.root.name,
                        "status": json.loads(
                            (self.root / "candidate/status.json").read_text()
                        ),
                        "plan": json.loads(
                            (self.root / "candidate/plan.json").read_text()
                        ),
                    }
                ),
                content_type="application/json",
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
    parser.add_argument("--joint", action="store_true")
    args = parser.parse_args()
    tensorboard_app = tensorboard.program.TensorBoard(
        server_class=lambda charts, flags: tensorboard.program.WerkzeugServer(
            Dashboard(args.root, charts, joint=args.joint), flags
        )
    )
    tensorboard_app.configure(
        argv=[
            "tensorboard",
            "--logdir",
            str(args.root / ("dashboard" if args.joint else "tensorboard")),
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
