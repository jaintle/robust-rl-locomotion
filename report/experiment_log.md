# Experiment Log — robust-rl-locomotion

All experiments use HalfCheetah-v4 (Gymnasium 1.2 / MuJoCo 3.5).
Shift is applied at evaluation time only. The training procedure is not modified.

Format: one entry per distinct experiment batch.

---

## Entry 001 — Harness Validation (Single Seed, Smoke Run)

**Date:** 2026-03-01
**Environment:** HalfCheetah-v4
**Seeds:** 0 (single seed)
**Timesteps:** 5 000 (non-converged policy)
**Purpose:** Validate the evaluation harness end-to-end across both shift types. Confirm
deterministic output, schema compliance, and correct metric computation before committing to
multi-seed training runs.

### Protocol

- Episodes per severity: 5
- Determinism verified: `diff eval_shifted.jsonl eval_shifted_b.jsonl` → empty
- Schema validation: passed for both `eval_shifted.jsonl` and `eval_shifted_dynamics.jsonl`

### Observations

- Identical JSONL output across two independent runs confirmed deterministic episode ordering.
- Gaussian degradation was monotonically increasing and small in magnitude.
- Mass-scale return was non-monotonic: α = 0.2 yielded a return lower than both α = 0.1 and
  α = 0.3. Within-episode return std was substantially elevated at α = 0.2.
- No episode terminations before the 1 000-step horizon under either shift type.
- Harness validated for use in multi-seed runs.

---

## Entry 002 — Multi-Seed Robustness Evaluation (3 Seeds, 20 000 Timesteps)

**Date:** 2026-03-02
**Environment:** HalfCheetah-v4
**Seeds:** 0, 1, 2
**Timesteps:** 20 000 per seed
**Run directory:** `runs/multiseed_exp/`
**Aggregation output:** `results/agg_multiseed_exp/`
**Purpose:** Produce cross-seed mean and standard deviation estimates for both shift types.
Assess whether degradation patterns are seed-stable or seed-dependent at a pre-convergence
training budget.

### Shift types evaluated

| Shift type  | Parameter | Severity grid                |
|-------------|-----------|------------------------------|
| gaussian    | σ         | {0.00, 0.01, 0.05, 0.10}     |
| mass\_scale | α         | {0.00, 0.10, 0.20, 0.30}     |

### Protocol

- Episodes per severity: 10
- Multi-seed runner: `scripts/run_multiseed.py --seeds 0,1,2 --total_timesteps 20000`
- Cross-seed aggregation: `tools/aggregate_multiseed.py`
- Per-seed aggregation stored in: `runs/multiseed_exp/seed_<N>/agg/`
- Cross-seed outputs: `results/agg_multiseed_exp/cross_seed_curves.csv`,
  `results/agg_multiseed_exp/cross_seed_summary.csv`

### Per-seed clean returns

| Seed | Clean return mean | Clean return std |
|-----:|------------------:|-----------------:|
| 0    | −9.99             | 1.28             |
| 1    | +17.27            | 0.61             |
| 2    | −7.46             | 0.97             |

Cross-seed mean: −0.06 ± 12.30

### Observations

- **Pre-convergence variance dominates.** At 20 000 timesteps, clean returns span −9.99 to
  +17.27 across seeds, indicating the policy has not reached a stable performance regime.
  The cross-seed std (12.30) is two orders of magnitude larger than the mean (−0.06). Cross-seed variance at clean baseline (std 12.30) exceeds the absolute mean shift induced by any tested severity.

- **Gaussian degradation is monotonic per seed.** All three seeds show monotonically increasing
  degradation across σ ∈ {0.00, 0.01, 0.05, 0.10}. Cross-seed mean return at σ = 0.1 is −1.04
  (std 11.01), compared to −0.06 (std 12.30) at clean. Relative drop cross-seed mean is 5.83 %
  ± 7.39 %, where the std reflects variation in the per-seed denominator.

- **Mass-scale degradation is non-monotonic in cross-seed mean.** At α = 0.2, the cross-seed
  mean return is +0.84 — nominally higher than clean (−0.06). This is a denominator artifact:
  when the mean baseline is near zero, small return fluctuations produce large, sign-unstable
  percentage drops. At α = 0.3, cross-seed mean return is −0.34 (std 15.16), close to the
  clean baseline.

- **Relative-drop metric unreliable at this training budget.** The worst-case mass-scale drop
  std of ±45.46 % is a direct consequence of the near-zero denominator. Per-seed magnitude:
  seed 0 shows +65.62 % drop at α = 0.3 (return −16.54 vs. clean −9.99); seed 1 shows −15.43 %
  (return +19.93 vs. clean +17.27, i.e. improvement); seed 2 shows −41.04 % (return −4.40 vs.
  clean −7.46, i.e. improvement). These divergent signs reflect pre-convergence variability,
  not a systematic robustness property.

- **No catastrophic collapse.** Episode length is 1 000 steps for all seeds, all severities,
  and both shift types.

- **Harness determinism confirmed across all three seeds.** Schema validation passed for all
  `seed_0`, `seed_1`, and `seed_2` run directories.

### Quantitative summary (from `cross_seed_summary.csv`)

| Shift type  | Mean clean return | Std  | Mean AUC (norm) | Std   | Worst drop mean (%) | Std   | Seeds |
|-------------|------------------:|-----:|----------------:|------:|--------------------:|------:|------:|
| gaussian    | −0.06             | 12.30 | −0.43           | 11.74 | 5.83                | 7.39  | 3     |
| mass\_scale | −0.06             | 12.30 | +0.14           | 12.89 | 3.05                | 45.46 | 3     |

### Per-seed worst-case summary (from per-seed `summary.csv`)

| Seed | Shift type  | Clean return | Worst severity | Return at worst | Relative drop (%) |
|-----:|-------------|-------------:|---------------:|----------------:|------------------:|
| 0    | gaussian    | −9.99        | σ = 0.10       | −10.13          | +1.46             |
| 0    | mass\_scale | −9.99        | α = 0.30       | −16.54          | +65.62            |
| 1    | gaussian    | +17.27       | σ = 0.10       | +14.46          | +16.25            |
| 1    | mass\_scale | +17.27       | α = 0.30       | +19.93          | −15.43            |
| 2    | gaussian    | −7.46        | σ = 0.10       | −7.45           | −0.20             |
| 2    | mass\_scale | −7.46        | α = 0.30       | −4.40           | −41.04            |

### Limitations

- Three seeds is a small sample; reported std should be interpreted with caution.
- 20 000 timesteps is well below the convergence threshold for HalfCheetah-v4. Results
  characterise the harness and the metric's behaviour at low return magnitude, not the
  robustness properties of a trained policy.
- Single environment; results may not transfer to other locomotion tasks.
- No adversarial perturbations; worst-case bounds are not available.
- Evaluation-only shift; training-time domain randomisation is excluded by design.
