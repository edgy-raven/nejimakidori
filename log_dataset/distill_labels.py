"""Cache phase-conditioned greedy distill policy labels beside an expert corpus."""

import argparse
import collections
import gzip
import hashlib
import json
import pathlib
import threading
import time

import numpy
import tensorflow as tf
from log_dataset import archives, features, replay, shanten
from training import distill_policy, hybrid


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def record_key(example):
    fields = example.features.feature
    return (
        fields["meta/game_id"].bytes_list.value[0].decode(),
        fields["meta/kyoku_id"].bytes_list.value[0].decode(),
        fields["meta/decision_index"].int64_list.value[0],
        fields["meta/actor"].int64_list.value[0],
    )


class Teacher:
    def __init__(self, root, device, capacity, *, cuda_graphs=False):
        opponent = distill_policy.load_engine(pathlib.Path(root), device)
        if cuda_graphs:
            opponent.brain = distill_policy.GraphBrain(opponent.brain)
        opponent.enable_quick_eval = False
        self.batcher = distill_policy.BatchedReactions(opponent, capacity)
        self.inference = opponent.react_batch
        opponent.react_batch = self.infer_phase
        self.opponent = opponent
        self.context = threading.local()

    def infer_phase(self, obs, masks, invisible):
        masks = [mask.copy() for mask in masks]
        allowed = numpy.ones(46, dtype=bool)
        if self.context.phase == "DISCARD":
            allowed[37:] = False
        elif self.context.phase == "RIICHI":
            allowed[38:] = False
        elif self.context.phase == "RESPONSE":
            allowed[:38] = False
            allowed[43:45] = False
        elif self.context.phase == "KAN":
            allowed[43:45] = False
        # The optional first request selects the tile for a kan.
        masks[-1] &= allowed
        assert masks[-1].any(), self.context.phase
        return self.inference(obs, masks, invisible)

    def game(self, game, requested):
        import libriichi.mjai

        try:
            raw = archives.open_archive(game["archive"]).read(game["member"])
            if raw.startswith(b"\x1f\x8b"):
                raw = gzip.decompress(raw)
            events = [json.loads(line) for line in raw.splitlines()]
            starts = [
                i
                for i, event in enumerate(events)
                if event["type"] == "start_kyoku"
            ]
            result = {}
            for start, end in zip(starts, starts[1:] + [len(events)]):
                state = replay.Replay(game["game_id"], events[start:end])
                bots = {}
                private_events = {}
                for index, actor, phase, _, _ in state.policy_decisions():
                    key = (
                        game["game_id"],
                        state.kyoku_id,
                        state.decision_index,
                        actor,
                    )
                    if key not in requested:
                        continue
                    assert requested[key] == features.PHASES[phase]
                    self.context.phase = phase
                    if (actor, phase) not in bots:
                        bots[actor, phase] = (
                            libriichi.mjai.Bot(self.opponent, actor),
                            0,
                        )
                    bot, cursor = bots[actor, phase]
                    if actor not in private_events:
                        private_events[actor] = []
                        for event in state.events:
                            event = dict(event)
                            if event["type"] == "start_kyoku":
                                event["tehais"] = [
                                    hand if seat == actor else ["?"] * 13
                                    for seat, hand in enumerate(event["tehais"])
                                ]
                            elif (
                                event["type"] == "tsumo"
                                and event["actor"] != actor
                            ):
                                event["pai"] = "?"
                            private_events[actor].append(json.dumps(event))
                    # Serialize each private event once per seat and hand.
                    # Only the observed prefix is ever sent to the teacher.
                    pending = private_events[actor][cursor : index + 1]
                    if state.events[index]["type"] == "dora":
                        bot = libriichi.mjai.Bot(self.opponent, actor)
                        trigger = index
                        while state.events[trigger]["type"] == "dora":
                            trigger -= 1
                        pending = (
                            private_events[actor][:trigger]
                            + private_events[actor][trigger + 1 : index + 1]
                            + [private_events[actor][trigger]]
                        )
                    for event in pending[:-1]:
                        bot.react(event, can_act=False)
                    response = json.loads(bot.react(pending[-1]))
                    bots[actor, phase] = (bot, index + 1)
                    if phase == "WIN":
                        choice = int(response["type"] == "hora")
                    elif phase == "KAN":
                        choice = (
                            1
                            + shanten.base_id(
                                response["consumed"][0]
                                if response["type"] == "ankan"
                                else response["pai"]
                            )
                            if response["type"] in ("ankan", "kakan")
                            else 0
                        )
                    elif phase == "DISCARD":
                        assert response["type"] == "dahai", response
                        choice = shanten.mjai_tile_code(response["pai"])
                    elif phase == "RIICHI":
                        declared = response["type"] == "reach"
                        if declared:
                            self.context.phase = "DISCARD"
                            response = json.loads(
                                bot.react(json.dumps(response))
                            )
                            del bots[actor, phase]
                        assert response["type"] == "dahai", response
                        choice = 37 * declared + shanten.mjai_tile_code(
                            response["pai"]
                        )
                    else:
                        if response["type"] == "none":
                            choice = 0
                        elif response["type"] == "daiminkan":
                            choice = 149
                        else:
                            assert response["type"] in ("chi", "pon"), response
                            offset = (
                                1
                                + 37
                                * features.chii_pattern(
                                    response["pai"], response["consumed"]
                                )
                                if response["type"] == "chi"
                                else 112
                            )
                            self.context.phase = "DISCARD"
                            cut = json.loads(bot.react(json.dumps(response)))
                            assert cut["type"] == "dahai", cut
                            choice = offset + shanten.mjai_tile_code(cut["pai"])
                            del bots[actor, phase]
                    result[key] = choice
            assert result.keys() == requested.keys(), (
                game["game_id"],
                requested.keys() - result.keys(),
            )
            return result
        finally:
            self.batcher.queue.put(None)


def run(args):
    tf.config.set_visible_devices([], "GPU")
    source = pathlib.Path(args.data).resolve()
    root = pathlib.Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    frozen = pathlib.Path(args.teacher).resolve()
    plan = {
        "data": str(source),
        "dataset_sha256": hashlib.sha256(
            (source / "dataset_summary.json").read_bytes()
        ).hexdigest(),
        "teacher": str(frozen),
        "teacher_sha256": {
            str(path.relative_to(frozen)): digest(path)
            for path in [
                frozen / "checkpoint",
                *sorted((frozen / "source").glob("*")),
            ]
            if path.is_file()
        },
        "selection": "greedy_within_decision_phase",
    }
    if (root / "plan.json").exists():
        assert json.loads((root / "plan.json").read_text()) == plan
    else:
        (root / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    games = {
        row["game_id"]: row
        for row in map(
            json.loads, (source / "games.jsonl").read_text().splitlines()
        )
    }
    teacher = Teacher(
        frozen, args.device, args.workers, cuda_graphs=args.cuda_graphs
    )
    paths = sorted(source.glob("train-*.tfrecord.gz"))
    if args.limit_shards:
        paths = paths[: args.limit_shards]
    writer = tf.summary.create_file_writer(
        str(
            pathlib.Path(args.tensorboard or root / "tensorboard")
            / f"worker-{args.shard_offset}"
        )
    )
    positions = 0
    started = time.monotonic()
    try:
        for path in paths[args.shard_offset :: args.shard_stride]:
            destination = root / (path.name + ".npy")
            if (root / (path.name + ".json")).exists():
                saved = json.loads((root / (path.name + ".json")).read_text())
                assert digest(path) == saved["source_sha256"]
                assert digest(destination) == saved["labels_sha256"]
                continue
            keys = []
            requested = collections.defaultdict(dict)
            for raw in tf.data.TFRecordDataset(
                str(path), compression_type="GZIP"
            ):
                example = tf.train.Example.FromString(bytes(raw.numpy()))
                key = record_key(example)
                keys.append(key)
                requested[key[0]][key] = int.from_bytes(
                    example.features.feature[
                        "feature/decision_phase"
                    ].bytes_list.value[0],
                    signed=True,
                )
            results = {}
            names = list(requested)
            # Populate the shared ZIP cache before threads request its index.
            for archive in {games[name]["archive"] for name in names}:
                archives.open_archive(archive)
            for begin in range(0, len(names), args.workers):
                futures = [
                    teacher.batcher.pool.submit(
                        teacher.game, games[name], requested[name]
                    )
                    for name in names[begin : begin + args.workers]
                ]
                for result in teacher.batcher.collect(futures):
                    results.update(result)
            choices = numpy.array(
                [results[key] for key in keys], dtype=numpy.int16
            )
            checked = 0
            for raw in tf.data.TFRecordDataset(
                str(path), compression_type="GZIP"
            ).batch(256):
                hybrid.decode(
                    raw,
                    tf.convert_to_tensor(
                        choices[checked : checked + len(raw)], tf.int32
                    ),
                )
                checked += len(raw)
            assert checked == len(choices)
            with destination.with_suffix(".partial").open("wb") as stream:
                numpy.save(stream, choices)
            destination.with_suffix(".partial").replace(destination)
            manifest = root / (path.name + ".json")
            manifest.with_suffix(".partial").write_text(
                json.dumps(
                    {
                        "source_sha256": digest(path),
                        "positions": len(choices),
                        "labels_sha256": hashlib.sha256(
                            destination.read_bytes()
                        ).hexdigest(),
                    },
                    indent=2,
                )
                + "\n"
            )
            manifest.with_suffix(".partial").replace(manifest)
            positions += len(choices)
            with writer.as_default(step=positions):
                tf.summary.scalar("labels/completed_positions", positions)
                tf.summary.scalar(
                    "labels/positions_per_second",
                    positions / (time.monotonic() - started),
                )
            writer.flush()
            print(
                json.dumps({"shard": path.name, "positions": len(keys)}),
                flush=True,
            )
    finally:
        teacher.batcher.pool.shutdown()
        writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--teacher", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--limit-shards", type=int)
    parser.add_argument("--shard-offset", type=int, default=0)
    parser.add_argument("--shard-stride", type=int, default=1)
    parser.add_argument("--tensorboard")
    parser.add_argument("--cuda-graphs", action="store_true")
    run(parser.parse_args())
