"""New public games are reviewed once and delivered despite interrupted posts."""

import http.server
import json
import threading
import time
import urllib.parse

import pytest
import requests

from bots import track_player


def test_poll_baselines_reviews_and_recovers_a_delivered_post(
    tmp_path, monkeypatch
):
    now = int(time.time())
    existing = {"uuid": "old", "startTime": now - 60, "players": []}
    games = [existing]
    posts = []
    reviews = []

    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def send_json(self, value):
            body = json.dumps(value).encode() + b"\n"
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            path = urllib.parse.urlparse(self.path)
            if path.path.startswith("/amae/"):
                assert "Authorization" not in self.headers
                assert urllib.parse.parse_qs(path.query)["mode"] == [
                    "8,9,11,12,15,16"
                ]
                self.send_json(games)
            elif path.path == "/discord/users/@me":
                assert self.headers["Authorization"] == "Bot test-token"
                self.send_json({"id": "bot"})
            else:
                assert path.path == "/discord/channels/channel/messages"
                self.send_json(posts)

        def do_POST(self):
            body = json.loads(
                self.rfile.read(int(self.headers["Content-Length"]))
            )
            if self.path == "/api/replay":
                assert "Authorization" not in self.headers
                assert body["seat"] == "auto"
                reviews.append(body)
                self.send_json(
                    {
                        "type": "complete",
                        "game": {
                            "seat": 2,
                            "players": [{"accountId": 42, "seat": 2}],
                            "grades": [
                                None,
                                None,
                                {"score": 91.25, "graded": 86},
                            ],
                        },
                    }
                )
            else:
                assert self.path == "/discord/channels/channel/messages"
                assert body["allowed_mentions"] == {
                    "parse": [],
                    "replied_user": False,
                }
                assert body["message_reference"]["message_id"] == "anchor"
                assert body["enforce_nonce"]
                posts.append(
                    {
                        "id": str(((now * 1000 - 1420070400000) << 22) + 1),
                        "author": {"id": "bot"},
                        "content": body["content"],
                    }
                )
                # Discord accepted it, but the caller did not get the receipt.
                self.close_connection = True

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        url = f"http://127.0.0.1:{server.server_port}"
        (tmp_path / "token").write_text("test-token")
        settings = {
            "amae_url": url + "/amae",
            "review_url": url,
            "discord_url": url + "/discord",
            "player_id": 42,
            "nickname": "tracked",
            "channel_id": "channel",
            "guild_id": "guild",
            "reply_to": "anchor",
            "token_path": str(tmp_path / "token"),
        }
        config = tmp_path / "config.json"
        config.write_text(json.dumps(settings))
        monkeypatch.setattr(
            "sys.argv", ["track_player", str(config), "--initialize"]
        )
        track_player.run()
        state_path = tmp_path / "state.json"
        assert json.loads(state_path.read_text())["seen"] == {"old": None}
        assert not posts
        games.append(
            {
                "uuid": "new-game",
                "startTime": now,
                "players": [
                    {"accountId": 11, "score": 35000},
                    {"accountId": 42, "score": 32000},
                    {"accountId": 12, "score": 18000},
                    {"accountId": 13, "score": 15000},
                ],
            }
        )
        with pytest.raises(requests.ConnectionError):
            track_player.poll(settings, state_path)
        assert "new-game" in json.loads(state_path.read_text())["pending"]
        assert "91.25/100" in posts[0]["content"]
        assert "Placement: 2/4" in posts[0]["content"]
        assert "new-game_a" in reviews[0]["url"]
        track_player.poll(settings, state_path)
        track_player.poll(settings, state_path)
        state = json.loads(state_path.read_text())
        assert state["seen"]["new-game"] == posts[0]["id"]
        assert not state["pending"]
        assert len(posts) == len(reviews) == 1
    finally:
        server.shutdown()
        server.server_close()
        worker.join()
