# Threat Model — robust-rl-locomotion v1.0

This document defines the perturbation model used in this repository, the scope of the evaluation
study, and what is explicitly excluded. It is intended to prevent misinterpretation of results
and to support comparison with other robustness evaluations.

---

## Overview

This repository evaluates the sensitivity of a trained PPO policy to two categories of
deployment-time distribution shift. The perturbations are non-adversarial, non-adaptive, and
applied exclusively at evaluation time. The threat model is motivated by practical deployment
scenarios: sensor noise from imperfect hardware and parameter mismatch between simulation and
physical systems (sim2real gap).

No claim is made about worst-case robustness, certified bounds, or resistance to adversarial
attack.

---

## Threat Categories

### Category 1 — Sensor Corruption (Observation Noise)

**Motivation.** Physical sensors introduce noise through quantisation, calibration drift, thermal
variation, and communication latency. At inference time, the policy receives a corrupted observation
rather than the true environment state.

**Model.** Additive i.i.d. Gaussian noise applied to the full observation vector at every step:

```
obs_perturbed(t) = obs(t) + ε(t),    ε(t) ~ N(0, σ²I)
```

**Injection point.** After the environment step, before the observation reaches the policy:
`env → perturbation → policy`.

**Parameters.**

| Property         | Value                                   |
|------------------|-----------------------------------------|
| Distribution     | Gaussian, mean zero, isotropic          |
| Severity (σ)     | {0.00, 0.01, 0.05, 0.10}               |
| Observation space | Normalised (as returned by env)        |
| Correlation      | None (i.i.d. across steps and dims)     |
| Adversarial      | No                                      |

**What this does not model.** Structured sensor failure (e.g., stuck sensor, complete channel
dropout), temporally-correlated noise (e.g., drift), adversarially-chosen noise, or sensor
latency effects.

---

### Category 2 — Parameter Mismatch (Sim2Real Gap)

**Motivation.** Simulation environments do not perfectly replicate physical systems. Body masses,
friction coefficients, damping, and actuator properties in the simulator deviate from their
real-world counterparts. This category models the inertial component of that mismatch.

**Model.** At the start of each episode, all MuJoCo body masses are scaled by a scalar drawn
uniformly from `[1 − α, 1 + α]`:

```
mass_scale ~ Uniform(1 − α, 1 + α)
body_mass_effective = body_mass_nominal × mass_scale
```

The same scalar is applied to all bodies; body-relative mass ratios are preserved.

**Injection point.** At episode reset, modifying the MuJoCo model before any action is taken:
`env.reset() → mass perturbation → episode rollout`.

**Parameters.**

| Property         | Value                                   |
|------------------|-----------------------------------------|
| Distribution     | Uniform, symmetric around 1.0           |
| Severity (α)     | {0.00, 0.10, 0.20, 0.30}               |
| Scope            | All body masses (scalar, not per-body)  |
| Physical range   | α ≤ 0.5 enforced (prevents zero mass)  |
| Adversarial      | No                                      |
| Per-episode      | Yes (new sample at each reset)          |

**What this does not model.** Per-body independent mass variation, friction perturbation,
damping perturbation, contact geometry shift, actuator gain variation, or time-varying parameter
drift during an episode.

---

## Scope of This Evaluation

### What Is Measured

- Mean episode return under each perturbation type and severity level.
- Within-episode return standard deviation (episode-level variability).
- Relative degradation: percentage return drop from the clean baseline.
- AUC over the severity grid (trapezoidal rule), as a scalar summary of degradation across the
  tested range.
- Cross-seed variance in the above metrics (3 seeds).

### What Is Not Covered

The following are explicitly excluded from v1.0:

| Excluded topic                        | Reason for exclusion                                           |
|---------------------------------------|----------------------------------------------------------------|
| Adversarial perturbations             | Gradient-based or search-based attacks; outside evaluation scope |
| Worst-case certified bounds           | No formal verification; requires different methodology         |
| Certified robustness (e.g., smoothing) | Distinct research direction                                   |
| Training-time domain randomisation   | By design; training is not modified                            |
| Real hardware validation              | Simulation-only study                                          |
| Multi-environment generalisation      | Single environment (HalfCheetah-v4) only                      |
| Partial observability / delays        | Full-state observation assumed; no latency modelled            |
| Reward function shift                 | Reward is unchanged between training and evaluation            |
| Per-body independent mass variation   | Uniform scalar scaling only                                    |
| Friction, damping, contact shifts     | Not implemented in v1.0                                        |
| Observation-space adversaries         | Non-adversarial noise only                                     |

---

## Relationship to Other Robustness Frameworks

This study is positioned as an evaluation protocol, not as an algorithm. It does not propose or
compare robust training methods (e.g., adversarial training, robust MDP solvers, domain
randomisation). The perturbations are designed to be representative of plausible deployment
shifts, not to be maximally challenging.

Results from this repository are appropriate as a baseline measurement of standard PPO under
mild distribution shift, not as a claim about robustness limits or guarantees.

---

## Version History

| Version | Shift types              | Notes                                          |
|---------|--------------------------|------------------------------------------------|
| v1.0    | gaussian, mass\_scale    | Initial release; evaluation-only perturbations |
