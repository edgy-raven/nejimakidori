"""Build the source catalog without loading TensorFlow or model weights."""

import argparse
import ast
import hashlib
import json
import math
import pathlib
import re
import xml.etree.ElementTree

import latex2mathml.converter
import numpy

from log_dataset import _native_collection, scoring
from model import actions, outcomes

ROOT = pathlib.Path(__file__).resolve().parents[2]
CATALOG = pathlib.Path(__file__).resolve().parent


def reference(path, symbol, sources, scope=None):
    lines = sources[path].splitlines()
    begin, end = 1, len(lines)
    if scope is not None:
        node = next(
            node
            for node in ast.parse(sources[path]).body
            if isinstance(node, (ast.ClassDef, ast.FunctionDef))
            and node.name == scope
        )
        begin, end = node.lineno, node.end_lineno
    return {
        "path": path,
        "line": next(
            index
            for index, line in enumerate(lines[begin - 1 : end], begin)
            if symbol in line
        ),
    }


def output_contract(sources):
    """Read the exact validated serving dictionary from the inference client."""
    tree = ast.parse(sources["model/inference.py"])
    definition = next(
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "expected"
            for target in node.targets
        )
    )
    return eval(
        compile(ast.Expression(definition), "model/inference.py", "eval"),
        {
            "__builtins__": {},
            "len": len,
            "actions": actions,
            "outcomes": outcomes,
            "scoring": scoring,
        },
    )


def build():
    annotations = json.loads((CATALOG / "annotations.json").read_text())
    paths = sorted(
        {
            *ROOT.glob("model/*.py"),
            *ROOT.glob("model/*.md"),
            *ROOT.glob("training/*.py"),
            *ROOT.glob("training/*.md"),
            *ROOT.glob("training/corrections/*.py"),
            *ROOT.glob("training/corrections/*.md"),
            *ROOT.glob("log_dataset/*.py"),
            *ROOT.glob("log_dataset/native_collection/cpp/*.cpp"),
            *ROOT.glob("log_dataset/native_collection/cpp/*.hpp"),
            ROOT / "log_dataset/native_collection/src/schema.rs",
            ROOT / "log_dataset/native_collection/src/features.rs",
            ROOT / "log_dataset/native_collection/src/lib.rs",
            ROOT / "log_dataset/native_collection/README.md",
        }
    )
    sources = {str(path.relative_to(ROOT)): path.read_text() for path in paths}
    schema = sources["log_dataset/native_collection/src/schema.rs"]
    declared = {
        name: {
            "dtype": dtype.lower(),
            "shape": [int(value) for value in shape.split(",") if value],
        }
        for name, dtype, shape in re.findall(
            r"^    (\w+): (\w+) \[([\d, ]*)\],$", schema, re.MULTILINE
        )
    }
    assert list(declared) == list(_native_collection.INPUTS)
    assert declared == _native_collection.INPUTS
    assert set(annotations["features"]) == set(declared)
    scoring_inputs = ast.literal_eval(
        next(
            node.value.args[0]
            for node in ast.parse(sources["model/encoder.py"]).body
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "SCORING_INPUTS"
                for target in node.targets
            )
        )
    )
    features = []
    offset = 0
    named_consumers = {name: [] for name in declared}
    for path, text in sources.items():
        if path.endswith(".py") and path.startswith(("model/", "training/")):
            for node in ast.walk(ast.parse(text)):
                if (
                    isinstance(node, ast.Constant)
                    and isinstance(node.value, str)
                    and node.value in named_consumers
                ):
                    named_consumers[node.value].append(
                        {
                            "path": path,
                            "line": node.lineno,
                            "role": "Named reference",
                        }
                    )
    for name, spec in declared.items():
        consumers = named_consumers[name]
        if name.startswith("river_"):
            consumers.append(
                {
                    **reference(
                        "model/encoder.py", "class RiverEncoding", sources
                    ),
                    "role": "Shared river stream",
                }
            )
        elif name not in scoring_inputs and name not in {
            "candidate_hand_counts",
            "candidate_hand_red",
            "candidate_melds",
            "candidate_hand_valid",
        }:
            consumers.append(
                {
                    **reference(
                        "model/encoder.py", "def encode_inputs", sources
                    ),
                    "role": "Public numeric board projection",
                }
            )
        producers = [
            {"path": path, "line": index}
            for path, text in sources.items()
            if path.endswith((".cpp", ".rs")) and not path.endswith("schema.rs")
            for index, line in enumerate(text.splitlines(), 1)
            if re.search(r"\b" + name + r"\b", line)
        ]
        elements = math.prod(spec["shape"])
        size = elements * numpy.dtype(spec["dtype"]).itemsize
        features.append(
            {
                "id": name,
                "name": name,
                "title": name.replace("_", " ").capitalize(),
                **spec,
                **annotations["features"][name],
                "elements": elements,
                "bytes": size,
                "offset": offset,
                "status": "active",
                "consumers": consumers,
                "producers": producers,
                "source": reference(
                    "log_dataset/native_collection/src/schema.rs",
                    name + ":",
                    sources,
                ),
            }
        )
        offset += size
    architecture = annotations["architecture"]
    for entry in [
        *architecture["nodes"],
        *architecture["blocks"],
        *architecture["block_edges"],
        *architecture["gradient_overlays"],
        *[phase["objective"] for phase in architecture["gradient_overlays"]],
        *[
            edge
            for phase in architecture["gradient_overlays"]
            for edge in phase["edges"]
        ],
        *[
            operation
            for block in architecture["blocks"]
            for operation in block["operations"]
        ],
        *annotations["training"],
        *annotations["mathematics"],
    ]:
        entry["source"] = reference(
            entry["source"]["path"],
            entry["source"]["symbol"],
            sources,
            scope=entry["source"].get("scope"),
        )
    for section in annotations["mathematics"]:
        for equation in section["equations"]:
            equation["mathml"] = latex2mathml.converter.convert(
                equation["latex"], display="block"
            )
        for symbol in section["symbols"]:
            symbol["mathml"] = latex2mathml.converter.convert(
                symbol["symbol"], display="inline"
            )
    architecture["modules"] = [
        {
            "id": path,
            "path": path,
            "description": ast.get_docstring(ast.parse(text)) or "",
            "symbols": [
                {
                    "name": node.name,
                    "kind": (
                        "class"
                        if isinstance(node, ast.ClassDef)
                        else "function"
                    ),
                    "description": ast.get_docstring(node) or "",
                    "source": {"path": path, "line": node.lineno},
                    "layers": [
                        ast.get_source_segment(text, call)
                        for call in ast.walk(node)
                        if isinstance(call, ast.Call)
                        and isinstance(call.func, ast.Attribute)
                        and call.func.attr
                        in {
                            "Dense",
                            "Conv1D",
                            "Embedding",
                            "MultiHeadAttention",
                            "LayerNormalization",
                        }
                    ],
                }
                for node in ast.parse(text).body
                if isinstance(node, (ast.ClassDef, ast.FunctionDef))
            ],
        }
        for path, text in sources.items()
        if path.startswith("model/") and path.endswith(".py")
    ]
    block_titles = {row["id"]: row["title"] for row in architecture["blocks"]}
    architecture["gradient_items"] = []
    for phase in architecture["gradient_overlays"]:
        block_titles[phase["objective"]["id"]] = phase["objective"]["title"]
        for edge in phase["edges"]:
            edge["id"] = "-".join(
                [phase["id"], edge["kind"], edge["from"], edge["to"]]
            )
            architecture["gradient_items"].append(
                {
                    **edge,
                    "phase": phase["id"],
                    "title": (
                        block_titles[edge["from"]]
                        + " → "
                        + block_titles[edge["to"]]
                    ),
                    "description": edge["label"],
                    "gradient": True,
                    "mathml": latex2mathml.converter.convert(
                        (
                            r"\frac{\partial\operatorname{stop}(x)}{\partial x}=0"
                            if edge["kind"] == "stop"
                            else r"\bar{x}_{\mathrm{path}}=J_f(x)^{\mathsf T}\bar{y}"
                        ),
                        display="block",
                    ),
                }
            )
    contract = output_contract(sources)
    assert set(contract) == set(annotations["outputs"])
    outputs = [
        {
            "id": name,
            "name": name,
            "title": name.replace("_", " ").capitalize(),
            "shape": shape,
            "dtype": "int32" if name == "critic_action_indices" else "float32",
            "description": annotations["outputs"][name],
            "row_axis": (
                "active candidate" if name.startswith("critic_") else "position"
            ),
            "signatures": ["serving_default"]
            + (["policy"] if name in actions.TRAIN_POLICY_HEADS else []),
            "source": reference(
                (
                    "model/actions.py"
                    if name in actions.SERVING_ACTOR_OUTPUTS
                    else "model/inference.py"
                ),
                '"' + name + '":',
                sources,
            ),
        }
        for name, shape in contract.items()
    ]
    for entry in annotations["auxiliaries"]:
        entry["source"] = reference(
            entry["source"]["path"], entry["source"]["symbol"], sources
        )
    source_files = [
        {
            "path": path,
            "text": text,
            "sha256": hashlib.sha256(text.encode()).hexdigest(),
        }
        for path, text in sources.items()
    ]
    digest = hashlib.sha256()
    for path in [
        CATALOG / "annotations.json",
        CATALOG / "build.py",
        CATALOG / "requirements.txt",
        *paths,
    ]:
        digest.update(str(path.relative_to(ROOT)).encode() + b"\0")
        digest.update(path.read_bytes())
    result = {
        "meta": {
            "title": "Nejimakidori model catalog",
            "scope": (
                "Current working source; not a production checkpoint inventory"
            ),
            "source_digest": digest.hexdigest(),
            "feature_count": len(features),
            "feature_elements": sum(row["elements"] for row in features),
            "feature_bytes": offset,
            "output_count": len(outputs),
            "description": (
                "Public tensors, shared representations, predictions "
                "and training flow."
            ),
        },
        "features": features,
        "groups": annotations["groups"],
        "architecture": architecture,
        "outputs": outputs,
        "auxiliaries": annotations["auxiliaries"],
        "training": annotations["training"],
        "mathematics": annotations["mathematics"],
        "vocabulary": annotations["vocabulary"],
        "source_files": source_files,
    }
    validate(result)
    return result


def validate(catalog):
    features = {row["name"]: row for row in catalog["features"]}
    assert set(features) == set(_native_collection.INPUTS)
    groups = {row["id"] for row in catalog["groups"]}
    for name, spec in _native_collection.INPUTS.items():
        assert features[name]["shape"] == spec["shape"]
        assert features[name]["dtype"] == spec["dtype"]
        assert features[name]["description"]
        assert features[name]["encoding"]
        assert features[name]["group"] in groups
    nodes = {node["id"] for node in catalog["architecture"]["nodes"]}
    for edge in catalog["architecture"]["edges"]:
        assert edge["from"] in nodes and edge["to"] in nodes
    blocks = catalog["architecture"]["blocks"]
    assert len({block["id"] for block in blocks}) == len(blocks)
    assert not nodes & {block["id"] for block in blocks}
    assert {block["parent"] for block in blocks} == nodes
    for block in blocks:
        assert block["parent"] in nodes
        assert block["summary"] and block["dimensions"]
        assert block["sharing"] and block["residual"]
        assert block["operations"]
        assert len({row["id"] for row in block["operations"]}) == len(
            block["operations"]
        )
        for operation in block["operations"]:
            assert operation["title"] and operation["description"]
            assert operation["dimensions"]
            assert operation["sharing"] and operation["residual"]
    block_ids = {block["id"] for block in blocks}
    block_edges = catalog["architecture"]["block_edges"]
    assert len({(edge["from"], edge["to"]) for edge in block_edges}) == len(
        block_edges
    )
    assert {
        edge[endpoint] for edge in block_edges for endpoint in ("from", "to")
    } == block_ids
    for edge in block_edges:
        assert edge["from"] != edge["to"]
        assert edge["label"]
        assert edge["kind"] in {
            "state",
            "features",
            "residual",
            "logits",
            "constraint",
        }
    pending = block_ids.copy()
    while pending:
        ready = {
            block
            for block in pending
            if not any(
                edge["to"] == block and edge["from"] in pending
                for edge in block_edges
            )
        }
        assert ready, "Architecture block graph contains a cycle"
        pending -= ready
    for section in catalog["mathematics"]:
        assert section["equations"] and section["symbols"]
        for equation in section["equations"]:
            assert equation["label"] and equation["latex"]
            assert all(ord(value) >= 32 for value in equation["latex"])
        for symbol in section["symbols"]:
            assert symbol["symbol"] and symbol["meaning"]
            assert all(ord(value) >= 32 for value in symbol["symbol"])
        for expression in [*section["equations"], *section["symbols"]]:
            assert (
                xml.etree.ElementTree.fromstring(expression["mathml"]).tag
                == "{http://www.w3.org/1998/Math/MathML}math"
            )
    assert len({row["id"] for row in catalog["mathematics"]}) == len(
        catalog["mathematics"]
    )
    topics = {row["id"]: row for row in catalog["mathematics"]}
    overlays = catalog["architecture"]["gradient_overlays"]
    assert {phase["id"] for phase in overlays} == {"actor", "critic"}
    assert len(overlays) == 2
    gradient_items = catalog["architecture"]["gradient_items"]
    assert len({row["id"] for row in gradient_items}) == len(gradient_items)
    assert len(gradient_items) == sum(len(p["edges"]) for p in overlays)
    for row in gradient_items:
        assert xml.etree.ElementTree.fromstring(row["mathml"]).tag == (
            "{http://www.w3.org/1998/Math/MathML}math"
        )
    for entry in [*blocks, *[phase["objective"] for phase in overlays]]:
        assert entry["equations"]
        for selection in entry["equations"]:
            assert selection["topic"] in topics
            assert selection["equations"]
            assert len(set(selection["equations"])) == len(
                selection["equations"]
            )
            for index in selection["equations"]:
                assert isinstance(index, int)
                assert 0 <= index < len(topics[selection["topic"]]["equations"])
    forward_pairs = {(edge["from"], edge["to"]) for edge in block_edges}
    for phase in overlays:
        assert phase["objective"]["id"] not in block_ids
        assert {row["block"] for row in phase["statuses"]} == block_ids
        assert len(phase["statuses"]) == len(block_ids)
        assert phase["description"] and phase["notes"]
        for row in phase["statuses"]:
            assert row["description"]
            assert row["status"] in {
                "trainable",
                "partial",
                "frozen",
                "target",
                "objective",
                "deterministic",
                "disconnected",
            }
        endpoints = block_ids | {phase["objective"]["id"]}
        assert len(
            {
                (edge["from"], edge["to"], edge["kind"])
                for edge in phase["edges"]
            }
        ) == len(phase["edges"])
        for edge in phase["edges"]:
            assert edge["from"] in endpoints and edge["to"] in endpoints
            assert edge["from"] != edge["to"]
            assert edge["kind"] in {"backward", "stop"}
            assert edge["label"]
            if edge["kind"] == "backward":
                assert edge["to"] != "native-input-contract"
                assert (
                    edge["from"] == phase["objective"]["id"]
                    or (edge["to"], edge["from"]) in forward_pairs
                )
    sources = {row["path"]: row["text"] for row in catalog["source_files"]}
    assert {row["name"] for row in catalog["outputs"]} == set(
        output_contract(sources)
    )
    for entry in [
        *catalog["features"],
        *catalog["outputs"],
        *catalog["auxiliaries"],
        *catalog["architecture"]["nodes"],
        *blocks,
        *block_edges,
        *overlays,
        *[phase["objective"] for phase in overlays],
        *[edge for phase in overlays for edge in phase["edges"]],
        *[operation for block in blocks for operation in block["operations"]],
        *catalog["training"],
        *catalog["mathematics"],
    ]:
        assert entry["source"]["path"] in sources
        assert (
            0
            < entry["source"]["line"]
            <= len(sources[entry["source"]["path"]].splitlines())
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument(
        "--output", type=pathlib.Path, default=CATALOG / "catalog.json"
    )
    args = parser.parse_args()
    text = json.dumps(build(), indent=2, ensure_ascii=False) + "\n"
    if args.check:
        assert (
            args.output.read_text() == text
        ), "Regenerate the catalog from current source"
    else:
        args.output.write_text(text)
    print(f"Validated catalog: {args.output}")


if __name__ == "__main__":
    main()
