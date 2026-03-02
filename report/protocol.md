# Evaluation Protocol — robust-rl-locomotion v1.0

This document describes the training protocol, evaluation protocol, perturbation definitions,
aggregation procedure, and determinism guarantees used in this repository. It is intended to
support independent reproduction and to serve as a reference for interpreting result files.

---

## 1. Training Protocol

All policies are trained with Proximal Policy Optimisation (PPO-Clip) on state observations.
Training uses a single environment (no vectorised environments). No perturbations, domain
randomisation, or curriculum learning are applied during training.

### Algorithm

| Component         | Implementation                                   |
|-------------------|--------------------------------------------------|
| Policy            | Gaussian actor with state-independent log\_std  |
| Critic            | Scalar value function                            |
| Advantage         | Generalised Advantage Estimation (GAE)           |
| Policy loss       | PPO-Clip: `max(ratio × A, clip(ratio, 1±ε) × A)` |
| Value loss        | Mean squared error                               |
| Entropy bonus     | Weighted entropy term (`ent_coef × H(π)`)        |
| Gradient clipping | L2 norm clipping                                 |

### Architecture

Both the actor and critic use a two-layer Tanh MLP trunk with 64 hidden units per layer.
All linear layers use orthogonal weight initialisation. The mean-output head of the actor uses
standard deviation 0.01; the value head uses 1.0. The log\_std is a global learnable parameter
(not observation-conditioned).

### Hyperparameters

| Hyperparameter        | Value              |
|-----------------------|--------------------|
| Total timesteps       | 1 000 000 (full)   |
| Rollout steps         | 1 024              |
| Update epochs         | 4                  |
| Minibatch size        | 256                |
| Learning rate         | 3 × 10⁻⁴ (Adam)   |
| Discount γ            | 0.99               |
| GAE λ                 | 0.95               |
| PPO clip coefficient ε | 0.2              |
| Entropy coefficient   | 0.0                |
| Value loss coefficient | 0.5               |
| Max gradient norm     | 0.5                |

### Seeding

At the start of each training run the following seeds are fixed:

```python
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
env.reset(seed=seed)
env.action_space.seed(seed)
env.observation_space.seed(seed)
```

The training seed is recorded in `config.json` and propagated to all downstream evaluation scripts.

### Saved Artifacts

Each training run produces the following files in `--save_dir`:

| File                 | Content                                         |
|----------------------|-------------------------------------------------|
| `config.json`        | All CLI arguments plus `obs_dim`, `action_dim`  |
| `train_summary.json` | Final training return statistics (last 100 eps) |
| `eval_clean.json`    | Deterministic clean evaluation result           |
| `metrics.csv`        | Per-update losses (policy, value, entropy, KL)  |
| `checkpoint.pt`      | `policy_state_dict`, `value_state_dict`, `optimizer_state_dict` |

---

## 2. Evaluation Protocol

All evaluation is deterministic. The policy is set to `eval()` mode and all actions are computed
as distribution means (`deterministic=True`), with no gradient computation (`torch.no_grad()`).

### Episode Seed List

A per-evaluation seed list is generated as:

```python
episode_seeds = [base_seed + 1000 + i for i in range(episodes)]
```

where `base_seed` equals the training seed for that run. This list is fixed before the evaluation
loop begins and is reused identically across all shift-type and severity combinations within a
single evaluation run. The seed list is hashed to a 64-character SHA-256 hex string:

```python
eval_seed_list_hash = sha256(",".join(str(s) for s in episode_seeds))
```

This hash is stored in every evaluation output file and validated by `tools/validate_results.py`.

### Clean Evaluation

Clean evaluation (`eval_clean.json`) is produced by `scripts/train_ppo_state.py` at the end of
training. It uses the deterministic policy and the fixed episode seed list, with no wrappers
applied.

### Shifted Evaluation

Shifted evaluation is run as a separate pass for each shift type. The same episode seed list is
reused across all severity values within the run, ensuring that hash values are consistent and
that severity differences are not confounded by episode ordering.

Shifted evaluation scripts accept `--episodes` as an argument. If the episode count matches the
clean evaluation, the `eval_seed_list_hash` values will be identical.

---

## 3. Gaussian Shift Definition

**Class:** `GaussianObsNoiseWrapper` (inherits `gymnasium.ObservationWrapper`)
**File:** `src/robust_rl_locomotion/envs/wrappers/obs_noise.py`

At each step, the wrapper applies:

```
obs_perturbed = obs + σ · ε,    ε ~ N(0, I)
```

where `σ` is the severity parameter and `ε` is sampled from a private `np.random.RandomState`
instance isolated from all global RNG state. No clipping is applied.

| Parameter          | Definition                                          |
|--------------------|-----------------------------------------------------|
| σ (sigma)          | Noise standard deviation, applied in observation space |
| Severity grid (v1) | σ ∈ {0.00, 0.01, 0.05, 0.10}                       |
| Noise seed         | `training_seed + 4242 + int(σ × 10⁶)`              |
| σ = 0.0            | No-op fast path; returns observation unchanged      |

---

## 4. Mass-Scale Shift Definition

**Class:** `MassScaleWrapper` (inherits `gymnasium.Wrapper`)
**File:** `src/robust_rl_locomotion/envs/wrappers/dynamics_shift.py`

At the start of each episode (i.e., on every `reset()` call), a scalar mass-scale factor is
sampled uniformly and applied to all MuJoCo body masses:

```
mass_scale ~ Uniform(1 − α, 1 + α)
body_mass[:] = base_body_mass × mass_scale
```

The base masses are copied at wrapper construction time and are never modified cumulatively.
The mass-scale factor uses a private, isolated `np.random.RandomState`.

| Parameter          | Definition                                          |
|--------------------|-----------------------------------------------------|
| α (alpha)          | Half-width of uniform mass-scale distribution       |
| Valid range        | 0.0 ≤ α ≤ 0.5 (enforced by ValueError)             |
| Severity grid (v1) | α ∈ {0.00, 0.10, 0.20, 0.30}                       |
| Mass seed          | `training_seed + 9000 + int(α × 10⁶)`              |
| last\_mass\_scale  | Exposed attribute; records the sampled scale factor |

The sampled `mass_scale` values are recorded per episode and their mean and standard deviation
are stored in `eval_shifted_dynamics.jsonl`.

---

## 5. Multi-Seed Aggregation

Three seeds (0, 1, 2) are trained and evaluated independently. For each seed, `tools/aggregate.py`
produces `curves.csv` and `summary.csv` under `<seed_dir>/agg/`.

`tools/aggregate_multiseed.py` discovers all `seed_*` subdirectories, runs per-seed aggregation
automatically if absent, then computes cross-seed statistics:

- For each `(shift_type, severity)`: mean and population standard deviation of `return_mean` and
  `relative_drop_pct` across seeds.
- For each `shift_type`: mean and population standard deviation of `clean_return_mean`,
  `auc_return_norm`, and `relative_drop_worst_pct` across seeds.

Population standard deviation (`ddof=0`) is used throughout. With `n=3` seeds, this is a
biased estimate; results should be interpreted accordingly.

---

## 6. Relative Degradation Formula

For a given shift type and severity *s*:

```
relative_drop_pct(s) = 100 × (R_clean − R(s)) / max(|R_clean|, 1 × 10⁻⁸)
```

where `R_clean` is `return_mean` from `eval_clean.json` and `R(s)` is `return_mean` from the
shifted evaluation at severity *s*. The denominator guard `1 × 10⁻⁸` prevents division by zero
for near-zero baselines. A positive value indicates degradation (lower return); a negative value
indicates improvement.

---

## 7. AUC Computation

The area under the return-vs-severity curve is computed using the trapezoidal rule over the
ordered severity grid `{s_0, s_1, …, s_n}`:

```
AUC = Σᵢ₌₀ⁿ⁻¹ (sᵢ₊₁ − sᵢ) × (R(sᵢ) + R(sᵢ₊₁)) / 2
```

The normalised AUC is:

```
AUC_norm = AUC / (s_max − s_min)    if s_max > s_min
AUC_norm = AUC                       otherwise
```

`AUC_norm` represents the mean return across the severity range, weighted by severity spacing. A
higher (less negative) `AUC_norm` indicates better average performance across the severity grid.

When fewer than two severity points are available, `AUC` and `AUC_norm` are reported as `NaN`.

---

## 8. Determinism and Reproducibility Guarantees

The following properties are guaranteed for any two invocations with identical arguments:

| Property                          | Mechanism                                              |
|-----------------------------------|--------------------------------------------------------|
| Identical episode trajectories    | Fixed per-episode seed list; deterministic env seeding |
| Identical JSONL output            | Verified by `diff` before committing results           |
| Identical CSV output              | All rows sorted by `(shift_type, severity)` before write |
| Cross-severity hash consistency   | Validated by `tools/validate_results.py`               |
| Wrapper RNG isolation             | Private `np.random.RandomState` per wrapper instance   |
| Global RNG integrity              | Wrapper RNG never touches global NumPy or Torch state  |

To verify determinism for any evaluation script, run the script twice with the same arguments
and compare outputs:

```bash
diff runs/seed_0/eval_shifted.jsonl runs/seed_0/eval_shifted_check.jsonl
# Expected: no output (identical files)
```

The `eval_seed_list_hash` field provides a lightweight integrity check: any two evaluation runs
that used the same episode seed list must produce the same hash value.
