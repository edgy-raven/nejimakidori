# Value experiments

These tools evaluate value estimates separately from the
[training schedule](README.md). They do not establish policy strength.

## Warm value-head continuation pilot
`python -m training.value_experiment collect --weights CHECKPOINT --output DATA`
collects one uniformly sampled decision per independently seeded base game,
then forks repeated continuations under the checkpoint's fixed stochastic
policy. Native forks resample unseen wall tiles and future actions while
retaining opponents' concealed hands. These targets reduce future-continuation
noise conditional on those hands; they are not hidden-hand posterior averages.
The target remains observed hand settlement with fixed ST3 continuation.

Each `roots-*.npz` shard stores root IDs, observable features, cached frozen
board embeddings, per-continuation returns and root actions, head and legality.
V and targets use the same current-hand projection in ST3 / 100 units:
`observed_payoffs` applies the fixed continuation mapping to the observed
current-hand ending. It never uses realized outcomes from later hands.
Evaluate predicted projected ST3 against the mean of these projected outcomes,
not the projection of an averaged settlement (the mapping is nonlinear).
All policy continuations contribute, including attempted calls that lose
priority. Q-like forecasts also condition on attempting the action: accepted
and preempted calls both contribute to training. Hypothetical accepted-call
features describe the candidate, while hard scoring constraints retain only
guaranteed pre-call facts. The planned post-call discard is contingent, so
its outcome belongs to the joint hand forecast. This V experiment does not
use Q-like forecasts or losses. Match strength remains a separate test.
It does not provide concealed hands to the value model. Unique contiguous root
IDs and whole-game 75/12.5/12.5 percent train/validation/test splits are required.

`python -m training.value_experiment fit --weights CHECKPOINT --data DATA
--output OUTPUT` compares unchanged V, single-return fitting and mean-return
fitting. Every arm restores the checkpoint's existing `value_hidden` and
`state_value` weights exactly; only Adam state starts fresh. Value fitting
leaves every non-value model weight unchanged. Validation mean-return error
selects the epoch (including epoch zero); test roots never participate in
fitting or selection. Reports distinguish mean-target error, individual-return
error, bias, explained variance, calibration and target sampling variance.

The same short AWR probe follows each arm, with identical sampled root actions,
observed returns, beta, learning rate and minibatch order. Shared encoder and
value weights stay fixed during these actor-head-only probes. This isolates
baseline changes and does not reproduce the full joint training recipe.
`*.value.weights.h5` saves the value-only fit; `*.policy.weights.h5` saves the
subsequent probe. Production matches are a separate validation stage. A lower
value error alone does not establish improved playing strength.

`sweep --weights CHECKPOINT --data DATA --split SPLIT --output OUTPUT`
searches learning rates and batch sizes on cached embeddings. Split values
are 0 for training, 1 for validation and 2 for excluded test games. The
unchanged checkpoint remains eligible. `confirm --compare CHECKPOINT` reports
paired root-bootstrap error differences on a separately collected panel.

`learning-curve` fits nested training subsets (default 2,048/4,096/8,192/12,288
roots) against the unchanged development validation split. It compares the
original warm head at LR 1e-6 with a zero-output residual correction of depth
2, dropout 0.1 at LR 1e-5, batch 64, across seeds 941/942/943. Subset order is
seeded and shared between arms; `--training-roots` overrides subset sizes.
Reports retain row membership, epoch histories, and training/validation errors.
This is a development diagnostic, not independent confirmation or a capacity
comparison at matched optimizer-update budgets.

`regularized --scoring-features` appends observable per-seat public meld fu
(divided by 128) and public bonus han (divided by 13) to the residual branch.
The learned V path still receives only its original board embedding and starts
unchanged. Compare with `regularized` using identical rates, penalties, batches
and fitting seeds. Use the same `--scoring-features` flag with `confirm-heads`
for these saved augmented-input heads. No concealed hands or outcome labels
enter either feature path.

`architecture` uses the same data/split arguments to compare widths
128/256/512 and additional residual branches with 1/2/3 GELU layers of width
128, with dropout 0 or 0.1. Widening copies the learned units and gives new
units zero output weights. Residual branches have zero output initialization.
Both preserve the original learned V at initialization. Residual variants
freeze the learned V path and train only its additive correction. Dropout applies only
to new residual hidden activations during fitting; validation is deterministic.
The original policy and encoder remain frozen.

Each configuration runs fitting seeds 941/942/943. Mean validation MSE across
seeds selects the configuration, and validation selects each seed's epoch.
`--residual-learning-rates` overrides the residual-branch rate grid (default
1e-6, 1e-5, 1e-4); the width-only control grid remains unchanged.
Selected heads are standalone `seed-941.keras`, `seed-942.keras` and
`seed-943.keras` models taking the frozen board embedding. They are not full
policy checkpoints. `confirm-heads --compare DIRECTORY` evaluates these three
heads against the unchanged learned V on fresh roots. Each seed must reduce
mean-target MSE by at least 2% and have a paired bootstrap upper confidence
bound below zero. `--alpha` sets the per-seed interval error rate. Independent
panels and repeated-search error allocation are controlled by the caller.

`encoder` uses the same data/split arguments to fit V through a separate
warm-started copy of the board encoder and the existing value layers. It
compares learning rates 1e-6/3e-6/1e-5 across fitting seeds 941/942/943, with
64-row batches and gradient clipping at norm 1. Unrelated head weights are
hash-checked unchanged. Validation selects epochs, including epoch zero, and
the mean validation error selects the learning rate. Test roots are excluded.
Saved `seed-*.value.weights.h5` files contain full-network weights for V
evaluation. Their policy heads share the adapted encoder, so these files are
not validated policy candidates. Collection must continue with the original
fixed policy, and any selected V needs independent fresh-game confirmation.
`confirm-encoder --compare DIRECTORY` evaluates these full-network V files on
fresh structured features using the same three-seed criteria as `confirm-heads`.
The original and adapted encoders both run with the same inference batch size.
Confirmation bootstraps use at least 10,000 replicates and increase with
stricter `--alpha` to retain at least 100 draws in each confidence tail;
root-index sampling is batched to limit memory use.

`regularized` searches a zero-initialized linear correction to the frozen
learned V, using L2 kernel penalties 0/1e-4/1e-3/1e-2/0.1/1.0. The correction
has 513 parameters on a 512-dimensional board embedding. The penalty affects
fitting only; unpenalized validation MSE selects epochs and configurations.
Use `--residual-learning-rates` to specify the optimization grid. Saved heads
use the same standalone `.keras` contract and `confirm-heads` evaluation.
