"""The fixed experiment recipe saves a reloadable serving model."""

import json
import pathlib
import subprocess

import numpy
import tensorflow as tf

from integration.rewrite import test_corpus_rebuild
from log_dataset import rebuild
from model import inference, model, objectives, records, replay, train


def test_validation_reserves_whole_games_independent_of_record_order(tmp_path):
    rows = [
        tf.train.Example(
            features=tf.train.Features(
                feature={
                    "meta/game_id": tf.train.Feature(
                        bytes_list=tf.train.BytesList(
                            value=[f"game-{game}".encode()]
                        )
                    )
                }
            )
        ).SerializeToString()
        for game in range(200)
        for _ in range(3)
    ]
    validation = records.game_partition(rows, "validation").numpy()
    training = records.game_partition(rows, "train").numpy()
    assert validation.any() and training.any()
    numpy.testing.assert_array_equal(training, ~validation)
    numpy.testing.assert_array_equal(
        validation.reshape(-1, 3),
        numpy.repeat(validation[::3, None], 3, axis=1),
    )
    numpy.testing.assert_array_equal(
        records.game_partition(rows[::-1], "validation"), validation[::-1]
    )
    events = json.loads(
        (
            pathlib.Path(__file__).parents[1]
            / "fixtures/model/pass_speed_round.json"
        ).read_text()
    )
    observation = next(replay.Replay("split", events).observations())
    example = tf.train.Example.FromString(
        rebuild.serialize_observation(observation, True)
    )
    with tf.io.TFRecordWriter(
        str(tmp_path / "train-00000.tfrecord.gz"), options="GZIP"
    ) as writer:
        for game in range(200):
            example.features.feature["meta/game_id"].bytes_list.value[:] = [
                f"game-{game}".encode()
            ]
            # A distinct weight lets the public dataset identify each game.
            example.features.feature["meta/sample_weight"].float_list.value[
                :
            ] = [float(game)]
            writer.write(example.SerializeToString())
    (tmp_path / "dataset_summary.json").write_text(
        json.dumps(records.DATA_CONTRACT)
    )
    for partition, selected in (
        ("train", training),
        ("validation", validation),
    ):
        weights = numpy.concatenate(
            [
                weights.numpy()
                for _, _, weights in records.dataset(
                    tmp_path, batch_size=32, partition=partition
                )
            ]
        )
        numpy.testing.assert_array_equal(
            numpy.sort(weights), numpy.flatnonzero(selected[::3])
        )


def test_deep_trunk_keeps_reference_budget_and_trains_every_block():
    events = json.loads(
        (
            pathlib.Path(__file__).parents[1]
            / "fixtures/model/pass_speed_round.json"
        ).read_text()
    )
    observations = list(replay.Replay("depth", events).observations())
    features, labels, _ = records.parse_batch(
        [rebuild.serialize_observation(row, True) for row in observations[:3]]
    )
    actor = model.build_model()
    assert abs(actor.count_params() / 23121122 - 1) < 0.01
    assert actor.get_layer("board_state").units == 512
    kernels = [
        actor.get_layer(f"board_block_{block}_{stage}").kernel
        for block in range(4)
        for stage in ("expand", "contract")
    ]
    with tf.GradientTape() as tape:
        outputs = actor(features, training=True)
        loss = objectives.policy_loss(labels, outputs)
    for gradient in tape.gradient(loss, kernels):
        assert gradient is not None
        assert numpy.isfinite(gradient.numpy()).all()
        assert float(tf.linalg.norm(gradient)) > 0


def test_experiment_training_exports_a_reloadable_saved_model(tmp_path):
    archives = tmp_path / "archives"
    archives.mkdir()
    test_corpus_rebuild.write_enriched_archive(archives)
    corpus = tmp_path / "corpus"
    run = tmp_path / "run"

    subprocess.run(
        test_corpus_rebuild.command(
            "-m",
            "log_dataset.rebuild",
            "--archives",
            str(archives),
            "--output",
            str(corpus),
            "--experiment",
        ),
        check=True,
    )
    subprocess.run(
        test_corpus_rebuild.command(
            "-m",
            "model.train",
            "--data",
            str(corpus),
            "--output",
            str(run),
            "--experiment",
        ),
        check=True,
    )

    predictor = inference.Predictor(run / "saved_model")
    features, _, _ = next(iter(records.dataset(corpus, batch_size=1)))
    result = predictor.infer(
        [{name: value[0].numpy() for name, value in features.items()}]
    )

    restored = train.GroupedModel(model.build_model())
    restored(features)
    restored.load_weights(run / "learner.weights.h5")
    assert restored.payoff_critic.units == restored.baseline.units == 2
    assert numpy.isfinite(restored.payoff_critic.kernel.numpy()).all()
    assert float(tf.linalg.norm(restored.payoff_critic.kernel)) > 0
    checkpoint = next((run / "top").glob("*/learner.weights.h5"))
    restored.load_weights(checkpoint)
    assert float(tf.linalg.norm(restored.payoff_critic.kernel)) > 0
    assert (run / "model.keras").is_file()
    assert (run / "saved_model" / "saved_model.pb").is_file()
    assert result.metadata["model_contract"] == "typed_payments"
    assert len(result.predictions) == 1
