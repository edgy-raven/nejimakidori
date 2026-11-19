# Record contract

[Build a dataset](README.md)

## Which positions enter training

The reserve is excluded first, then games are deterministically assigned to
training, validation and held-out partitions. Historical ranks select all 9+
dan seats, 75% of 8d and 50% of 7d seats. Other players still supply game context.
Forced-only decisions produce no training rows.

Restoration reads every retained training position once, including the final
partial batch. The reader uses natural train/validation partitions; the former
focused, placement-gap and mixed oversampling paths are removed. The MMD loop
selects targeted and normal roots from fresh matches independently of this
restoration corpus.

Validation uses a deterministic 1/16 decision sample from the natural validation
partition. It is stored in separate shards so validation can run without
scanning the training corpus.

[afk.py](afk.py) removes hands with suspected inactivity: consecutive long
all-tsumogiri hands, or a long unforced tsumogiri run that ignores improving
draws outside tenpai and opponent riichi. These are behavioral heuristics;
the archives have no timeout or disconnect signal. A flagged hand is removed
for all seats, while the remaining hands keep their original context.

## Features and labels

The [native engine](native_collection/README.md) extracts public features.
[replay.py](replay.py) reconstructs recorded decisions;
[training_targets.py](training_targets.py) and [scoring.py](scoring.py)
produce their targets. Hidden hands and later outcomes are available to label
generation, never to the policy inputs.
The correction sampler uses [sakigiri_regret.py](sakigiri_regret.py)
for root witnesses; these are not serialized restoration labels.

[records.py](records.py) defines label names, shapes and serialization.
Dataset consumers check its schema and scoring definitions against
`dataset_summary.json`. Rebuild in a new directory after a schema change;
editing an old summary does not convert its records.

The current feature schema has 127 tensors. It omits the eight
`discard_action_{normal,all}_upgrade_` fields, two fixed-reference scoring
tensors and five prior-only scoring tensors. Serialized feature maps and
packed native rows are generated from this schema. TensorFlow parsing ignores
extra serialized keys, but the dataset schema check does not accept an older
summary as the current contract. Rebuild records rather than relabeling an old
dataset summary. See the [native input contract](native_collection/README.md)
for checkpoint and exported-signature conversion requirements.

[training_data.py](training_data.py) stores counterfactual scoring labels
`critic/bonus` and `critic/score`, alongside observed terminal payments used by
the settlement likelihood. These outcomes are targets, never encoder inputs.
The only record metadata is game, hand, decision and actor provenance, plus the
game partition.

The current contract removes unused `sakigiri_pair`, `sakigiri_weight` and
`discard_regret` labels; constant sample weights; all focused-sampling flags;
and the twelve-value `critic/settlement` state used solely by the former
`projected_st3` diagnostic. The parser no longer computes that diagnostic or
unused focus and point summaries. `records.parse_batch` returns inputs and
labels; `training_data.parse_batch` additionally returns critic supervision.
Restoration action labels, including `continuation_discard`, remain intact.
Rebuild datasets in a new directory: old schema summaries are rejected, and
frozen source datasets are not rewritten in place.

For prediction meanings, see [the model architecture](../model/ARCHITECTURE.md).
