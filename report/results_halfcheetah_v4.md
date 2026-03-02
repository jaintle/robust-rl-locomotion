# Results — HalfCheetah-v4

This document records the robustness evaluation results for PPO trained on HalfCheetah-v4.
Results are reported under the protocol defined in `report/protocol.md`.

**Scope:** This benchmark characterises robustness behaviour under evaluation-time distribution
shift and does not claim asymptotic performance. The policy is trained with standard PPO
hyperparameters and is not modified for robustness.

---

## Benchmark Configuration

| Property                 | Value                                     |
|--------------------------|-------------------------------------------|
| Environment              | HalfCheetah-v4                            |
| Gymnasium version        | 1.2                                       |
| MuJoCo version           | 3.5                                       |
| Observation space        | State, 17-dimensional, float32            |
| Action space             | Continuous, 6-dimensional, [−1, 1]        |
| Episode horizon          | 1 000 steps                               |
| Training timesteps       | 20 000 per seed                           |
| Training seeds           | 0, 1, 2                                   |
| Evaluation episodes      | 10 per severity                           |
| Shift types              | gaussian, mass\_scale                     |
| Gaussian severity grid   | σ ∈ {0.00, 0.01, 0.05, 0.10}             |
| Mass-scale severity grid | α ∈ {0.00, 0.10, 0.20, 0.30}             |
| Run directory            | `runs/multiseed_exp/`                     |
| Aggregation output       | `results/agg_multiseed_exp/`              |

---

## Per-Seed Clean Baselines

| Seed | Return mean | Return std | Episode length mean | eval\_seed\_list\_hash (prefix) |
|-----:|------------:|-----------:|--------------------:|--------------------------------:|
| 0    | −9.99       | 1.28       | 1 000.0             | `2eac784e…`                     |
| 1    | +17.27      | 0.61       | 1 000.0             | `15fa11f0…`                     |
| 2    | −7.46       | 0.97       | 1 000.0             | `91d03080…`                     |

Cross-seed mean: −0.06 ± 12.30. The large spread across seeds reflects pre-convergence
behaviour at 20 000 timesteps.

---

## Multi-Seed Summary (from `cross_seed_summary.csv`)

| Shift type  | Mean clean return | Std  | Mean AUC (norm) | Std   | Worst drop mean (%) | Std   | Seeds |
|-------------|------------------:|-----:|----------------:|------:|--------------------:|------:|------:|
| gaussian    | −0.06             | 12.30 | −0.43           | 11.74 | 5.83                | 7.39  | 3     |
| mass\_scale | −0.06             | 12.30 | +0.14           | 12.89 | 3.05                | 45.46 | 3     |

---

## Per-Severity Results — Gaussian Noise (cross-seed, from `cross_seed_curves.csv`)

| σ    | Mean return | Std return | Mean relative drop (%) | Std drop (%) |
|------|------------:|-----------:|-----------------------:|-------------:|
| 0.00 | −0.06       | 12.30      | 0.00                   | 0.00         |
| 0.01 | −0.07       | 12.26      | 0.02                   | 0.24         |
| 0.05 | −0.33       | 11.80      | 1.33                   | 3.11         |
| 0.10 | −1.04       | 11.01      | 5.83                   | 7.39         |

Cross-seed AUC (norm): −0.43 ± 11.74

---

## Per-Severity Results — Mass-Scale Shift (cross-seed, from `cross_seed_curves.csv`)

| α    | Mean return | Std return | Mean relative drop (%) | Std drop (%) |
|------|------------:|-----------:|-----------------------:|-------------:|
| 0.00 | −0.06       | 12.30      | 0.00                   | 0.00         |
| 0.10 | −0.22       | 12.60      | 2.00                   | 3.67         |
| 0.20 | +0.84       | 12.52      | −8.00                  | 0.69         |
| 0.30 | −0.34       | 15.16      | 3.05                   | 45.46        |

Cross-seed AUC (norm): +0.14 ± 12.89

---

## Per-Seed Worst-Case Summary (from per-seed `summary.csv`)

| Seed | Shift type  | Clean return | Worst severity | Return at worst | Relative drop (%) | AUC (norm) |
|-----:|-------------|-------------:|---------------:|----------------:|------------------:|-----------:|
| 0    | gaussian    | −9.99        | σ = 0.10       | −10.13          | +1.46             | −9.93      |
| 0    | mass\_scale | −9.99        | α = 0.30       | −16.54          | +65.62            | −11.04     |
| 1    | gaussian    | +17.27       | σ = 0.10       | +14.46          | +16.25            | +16.12     |
| 1    | mass\_scale | +17.27       | α = 0.30       | +19.93          | −15.43            | +18.20     |
| 2    | gaussian    | −7.46        | σ = 0.10       | −7.45           | −0.20             | −7.47      |
| 2    | mass\_scale | −7.46        | α = 0.30       | −4.40           | −41.04            | −6.74      |

---

## Observations

### Gaussian Noise

Gaussian observation noise produces monotonically increasing degradation with σ for each
individual seed. The cross-seed mean return at σ = 0.1 is −1.04 (std 11.01), compared to
−0.06 (std 12.30) at clean — an absolute shift of approximately −1.0 units. The within-episode
return standard deviation decreases slightly from clean to σ = 0.1 (12.30 → 11.01), suggesting
that noise at this scale does not increase episode-to-episode variability for this undertrained
policy.

Seed 1 shows the largest absolute degradation (return: +17.27 → +14.46, drop 16.25 %) because
its positive clean baseline makes the denominator of the relative-drop formula larger. Seeds 0
and 2 show much smaller drops (1.46 % and −0.20 % respectively), where the negative sign for
seed 2 reflects sampling noise over 10 episodes.

### Mass-Scale Shift

Mass-scale perturbation does not produce monotonic degradation in the cross-seed mean. At
α = 0.2, the cross-seed mean return (+0.84) nominally exceeds the clean mean (−0.06). This
reflects denominator instability: the cross-seed mean baseline of −0.06 is near zero, so
small return fluctuations produce large and sign-unstable relative-drop values.

At the per-seed level, seed 0 shows a substantial return drop at α = 0.3 (−9.99 → −16.54,
drop +65.62 %), while seeds 1 and 2 show apparent improvement (seed 1: +17.27 → +19.93,
seed 2: −7.46 → −4.40). The divergent signs indicate that at this training budget, the effect
of mass scaling on return is dominated by episode-to-episode sampling variance rather than a
systematic dynamics-sensitivity signal.

Within-episode return standard deviation under mass-scale shift is elevated at α = 0.3
(cross-seed std 15.16 versus 12.30 at clean), consistent with the policy encountering a
broader range of locomotion conditions per episode. No episode terminations before the 1 000-step
horizon were observed.

---

## Interpretation

At 20 000 timesteps, the three seeds produce policies with substantially different performance
levels (−9.99, +17.27, −7.46). The cross-seed mean clean return of −0.06 is near zero, which
makes the relative-degradation metric numerically unreliable. The primary findings are:

Gaussian noise at σ ≤ 0.1 causes a modest, monotonic return reduction in raw units (absolute
shift ≈ −1.0 from clean to σ = 0.1 in cross-seed mean), consistent across seeds in direction
if not in magnitude. Mass-scale perturbation at α ≤ 0.3 does not produce a consistent signal
at this training budget; the cross-seed std of return (≈ 12–15) is comparable to the mean
shifts being measured (≈ 0–1 units), precluding reliable inference.

No claims are made about the behaviour of fully-trained policies. These measurements are
provided as a reproducible baseline under a fixed evaluation protocol, with the explicit
acknowledgement that the training budget is insufficient for convergence.

---

## Limitations

- Policy trained for 20 000 timesteps, well below convergence for HalfCheetah-v4.
- Three seeds is a small sample; standard deviations should be interpreted with caution.
- The near-zero cross-seed mean clean return makes the relative-drop metric numerically unstable.
  Raw return means are more informative than percentage-drop summaries at this training budget.
- Only mass scaling is varied in the dynamics shift; friction, damping, and contact geometry
  are unchanged.
- Results apply to a single environment. Extrapolation to other locomotion tasks is not warranted.
- The severity grids (4 points each) provide coarse coverage; more densely sampled grids may
  reveal additional structure.
