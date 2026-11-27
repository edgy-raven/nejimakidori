"""Score policy agreement with recorded expert and Mortal validation actions."""

import argparse
import hashlib
import json
import pathlib
import shutil
import tempfile
import time

import numpy
import tensorflow as tf

from log_dataset import records, training_data
from model import actions, actor, export
from training import monitor, restoration


def decision_totals(batch, log_policy, target, supported):
    correct = tf.argmax(log_policy, axis=1, output_type=target.dtype) == target
    ce = -tf.gather(log_policy, target, batch_dims=1)
    # The response distribution factors into branch mass and a post-call cut.
    branch_log = tf.stack(
        [
            tf.reduce_logsumexp(group, axis=1)
            for group in tf.split(log_policy, [1, 37, 37, 37, 37, 1], axis=1)
        ],
        axis=1,
    )
    branch = tf.where(
        target == 0, 0, tf.where(target == 149, 5, 1 + (target - 1) // 37)
    )
    cut_log = tf.gather(
        tf.reshape(log_policy[:, 1:149], [-1, 4, 37]),
        tf.clip_by_value(branch - 1, 0, 3),
        batch_dims=1,
    )
    response = batch["head"] == actions.TRAIN_POLICY_HEADS.index(
        "response_policy_logits"
    )
    accepted = response & (branch > 0) & (branch < 5)
    branch_match = tf.argmax(branch_log, 1, output_type=target.dtype) == branch
    cut_match = tf.argmax(cut_log, 1, output_type=target.dtype) == (
        (target - 1) % 37
    )
    values = (
        tf.ones_like(correct),
        correct,
        supported,
        tf.where(supported, ce, 0.0),
        response,
        response & branch_match,
        accepted,
        accepted & cut_match,
    )
    return tf.math.unsorted_segment_sum(
        tf.stack([tf.cast(value, tf.float64) for value in values], axis=1),
        batch["head"],
        len(actions.TRAIN_POLICY_HEADS),
    )


def rates(row):
    decisions, correct, supported, nll = row[:4]
    return {
        "decisions": int(decisions),
        "matches": int(correct),
        "errors": int(decisions - correct),
        "top1": correct / decisions if decisions else None,
        "supported_decisions": int(supported),
        "supported_nll": nll / supported if supported else None,
    }


def summarize(totals):
    result = {
        "all": rates(totals.sum(axis=0)),
        **{
            name.removesuffix("_logits"): rates(row)
            for name, row in zip(actions.TRAIN_POLICY_HEADS, totals)
        },
    }
    responses, branch_correct, accepted, cut_correct = totals[:, 4:].sum(axis=0)
    result["response_policy"].update(
        branch_top1=branch_correct / responses if responses else None,
        accepted_chi_pon_decisions=int(accepted),
        conditional_discard_top1=cut_correct / accepted if accepted else None,
    )
    return result


def run(args):
    if args.batch_size < 1 or (
        args.max_decisions is not None and args.max_decisions < 1
    ):
        raise ValueError("Batch size and decision limit must be positive")
    records.validate_summary(args.data)
    manifest = (args.data / "dataset_summary.json").read_bytes()
    summary = json.loads(manifest)
    assert summary["validation_sample"] == training_data.VALIDATION_SAMPLE
    paths = []
    count = 0
    for row in summary["shard_manifest"]:
        if row["validation_count"]:
            path = args.data / row["validation_path"]
            if "shard_readiness" in summary:
                receipt = args.data / (row["path"] + ".receipt.json")
                published = json.loads(receipt.read_text())
                if (
                    path.stat().st_size != published["validation_bytes"]
                    or row["validation_count"] != published["validation_count"]
                ):
                    raise ValueError(
                        f"Invalid published validation file {path}"
                    )
            paths.append(str(path))
            count += row["validation_count"]
            if args.max_decisions is not None and count >= args.max_decisions:
                break
    if not count:
        raise ValueError("Empty validation cohort")
    decisions = min(count, args.max_decisions or count)
    batches = (
        tf.data.TFRecordDataset(paths, compression_type="GZIP")
        .take(decisions)
        .batch(args.batch_size)
    )
    if args.weights:
        # The learner atomically replaces latest.weights.h5 as it saves.
        with tempfile.TemporaryDirectory() as tmp:
            snapshot = pathlib.Path(tmp) / "snapshot.weights.h5"
            shutil.copyfile(args.weights, snapshot)
            with snapshot.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            network = export.build_network(
                training_data.parse_batch(next(iter(batches))[:1])[0]
            )
            network.load_weights(snapshot)
        module = export.ServingModel(network)
        policy = module.policy
        checkpoint = args.weights.name
    else:
        module = tf.saved_model.load(str(args.model))
        policy = module.signatures["policy"]
        checkpoint = args.model.name
        # A standalone SavedModel need not carry the original H5 hash.
        digest = None

    @tf.function(input_signature=[tf.TensorSpec([None], tf.string)])
    def score(raw):
        parsed = tf.io.parse_example(
            raw,
            {
                name: tf.io.FixedLenFeature([], tf.string)
                for name in (*records.FIELDS, "expert/mortal_action")
            },
        )
        features, labels = records.decode_batch(parsed)
        head, expert, legal = restoration.policy_choices(features, labels)
        batch = {
            "features": features,
            "head": head,
            "expert": expert,
            "legal": legal,
            "mortal_action": tf.cast(
                tf.reshape(
                    tf.io.decode_raw(parsed["expert/mortal_action"], tf.int16),
                    [-1],
                ),
                tf.int32,
            ),
        }
        log_policy = actor.distribution(
            policy(**batch["features"]), batch["head"], batch["legal"]
        )
        return tf.stack(
            [
                decision_totals(
                    batch,
                    log_policy,
                    batch[name],
                    restoration.policy_target_supported(
                        batch["features"], batch["head"], batch[name]
                    ),
                )
                for name in ("expert", "mortal_action")
            ]
        )

    totals = numpy.zeros((2, len(actions.TRAIN_POLICY_HEADS), 8), numpy.float64)
    for raw in batches:
        totals += score(raw).numpy()
    if not numpy.isfinite(totals).all():
        raise ValueError("Nonfinite validation measurements")
    if not numpy.all(totals[:, :, 0].sum(axis=1) == decisions):
        raise ValueError("Validation record count disagrees with manifest")
    return {
        "checkpoint": checkpoint,
        "checkpoint_sha256": digest,
        "dataset_sha256": hashlib.sha256(manifest).hexdigest(),
        "partition": "validation",
        "max_decisions": args.max_decisions,
        "validation_shards": len(paths),
        "evaluated_at": time.time(),
        "expert": summarize(totals[0]),
        "mortal": summarize(totals[1]),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--model", type=pathlib.Path)
    source.add_argument("--weights", type=pathlib.Path)
    parser.add_argument("--data", type=pathlib.Path, required=True)
    parser.add_argument("--max-decisions", type=int)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--output", type=pathlib.Path)
    args = parser.parse_args()
    report = run(args)
    if args.output:
        monitor.write_json(args.output, report)
    else:
        print(json.dumps(report, indent=2, allow_nan=False))
