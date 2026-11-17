"""Explicitly add neutral river attention branches to existing weights."""

import argparse
import pathlib
import shutil
import tempfile

import h5py

from model import export


def initialize(weights, output, inputs):
    model = export.build_network(inputs)
    with tempfile.TemporaryDirectory() as directory:
        template = pathlib.Path(directory) / "template.weights.h5"
        model.save_weights(template)
        with h5py.File(weights) as original, h5py.File(
            template, "r+"
        ) as target:
            original_arrays = {}
            target_arrays = {}
            original.visititems(
                lambda name, obj: (
                    original_arrays.update({name: obj.shape})
                    if isinstance(obj, h5py.Dataset)
                    else None
                )
            )
            target.visititems(
                lambda name, obj: (
                    target_arrays.update({name: obj.shape})
                    if isinstance(obj, h5py.Dataset)
                    else None
                )
            )
            added = target_arrays.keys() - original_arrays.keys()
            if (
                not added
                or any(
                    not any(
                        branch in name
                        for branch in (
                            "/candidate_river_attention/",
                            "/river_attention/",
                        )
                    )
                    for name in added
                )
                or any(
                    target_arrays.get(name) != shape
                    for name, shape in original_arrays.items()
                )
            ):
                raise ValueError(
                    "Weights must differ only by missing river attention"
                )
            for name in original_arrays:
                target[name][...] = original[name][...]
        model.load_weights(template)
        with template.open("rb") as source, pathlib.Path(output).open(
            "xb"
        ) as dest:
            shutil.copyfileobj(source, dest)
    return model


def run():
    from training import self_play

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()
    initialize(args.weights, args.output, self_play.initial_features(941))


if __name__ == "__main__":
    run()
