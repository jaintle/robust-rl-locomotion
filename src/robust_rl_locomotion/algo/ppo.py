"""
Minimal CleanRL-style PPO for continuous control (state observations only).

Phase 3 — no perturbations, no pixel mode, no external RL frameworks.

Components:
  PolicyNetwork  – Gaussian actor with learnable log_std
  ValueNetwork   – scalar critic
  RolloutBuffer  – single-rollout storage + GAE
  PPOAgent       – thin wrapper exposing act() and update()
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _layer_init(layer: nn.Linear, std: float = np.sqrt(2), bias_const: float = 0.0) -> nn.Linear:
    """Orthogonal weight init (standard for PPO)."""
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


def _build_mlp(in_dim: int, hidden: tuple[int, ...]) -> tuple[nn.Sequential, int]:
    """Build a Tanh-MLP trunk. Returns (module, output_dim)."""
    layers: list[nn.Module] = []
    prev = in_dim
    for h in hidden:
        layers.append(_layer_init(nn.Linear(prev, h)))
        layers.append(nn.Tanh())
        prev = h
    return nn.Sequential(*layers), prev


# ---------------------------------------------------------------------------
# Networks
# ---------------------------------------------------------------------------

class PolicyNetwork(nn.Module):
    """Gaussian policy with state-independent log_std parameter."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden: tuple[int, ...] = (64, 64),
    ) -> None:
        super().__init__()
        trunk, out_dim = _build_mlp(obs_dim, hidden)
        self.trunk = trunk
        # Small std init for the mean head → early actions near zero.
        self.mean_head = _layer_init(nn.Linear(out_dim, action_dim), std=0.01)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def _dist(self, obs: torch.Tensor) -> torch.distributions.Normal:
        mean = self.mean_head(self.trunk(obs))
        std = self.log_std.exp().expand_as(mean)
        return torch.distributions.Normal(mean, std)

    def act(
        self, obs: torch.Tensor, deterministic: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        dist = self._dist(obs)
        action = dist.mean if deterministic else dist.rsample()
        logprob = dist.log_prob(action).sum(-1)
        return action, logprob

    def evaluate(
        self, obs: torch.Tensor, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute log-prob and entropy for a batch of (obs, actions)."""
        dist = self._dist(obs)
        logprob = dist.log_prob(actions).sum(-1)
        entropy = dist.entropy().sum(-1)
        return logprob, entropy


class ValueNetwork(nn.Module):
    """Scalar value function."""

    def __init__(self, obs_dim: int, hidden: tuple[int, ...] = (64, 64)) -> None:
        super().__init__()
        trunk, out_dim = _build_mlp(obs_dim, hidden)
        self.trunk = trunk
        self.head = _layer_init(nn.Linear(out_dim, 1), std=1.0)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        # Returns shape () for single obs, (batch,) for batched obs.
        return self.head(self.trunk(obs)).squeeze(-1)


# ---------------------------------------------------------------------------
# Rollout buffer
# ---------------------------------------------------------------------------

class RolloutBuffer:
    """Fixed-capacity buffer for one rollout of a single environment."""

    def __init__(
        self,
        rollout_steps: int,
        obs_dim: int,
        action_dim: int,
        device: torch.device,
    ) -> None:
        self.rollout_steps = rollout_steps
        self.device = device
        self.obs      = torch.zeros(rollout_steps, obs_dim,     device=device)
        self.actions  = torch.zeros(rollout_steps, action_dim,  device=device)
        self.logprobs = torch.zeros(rollout_steps,              device=device)
        self.rewards  = torch.zeros(rollout_steps,              device=device)
        self.dones    = torch.zeros(rollout_steps,              device=device)
        self.values   = torch.zeros(rollout_steps,              device=device)
        self.ptr = 0

    def reset(self) -> None:
        self.ptr = 0

    def store(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        logprob: torch.Tensor,
        reward: float,
        done: bool,
        value: torch.Tensor,
    ) -> None:
        self.obs[self.ptr]      = obs
        self.actions[self.ptr]  = action
        self.logprobs[self.ptr] = logprob
        self.rewards[self.ptr]  = reward
        self.dones[self.ptr]    = float(done)
        self.values[self.ptr]   = value
        self.ptr += 1

    def compute_gae(
        self,
        next_value: torch.Tensor,
        next_done: bool,
        gamma: float,
        gae_lambda: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Generalised Advantage Estimation over stored steps.

        Args:
            next_value: Bootstrap value for the state after the last step.
            next_done:  Whether the last step ended the episode.
            gamma:      Discount factor.
            gae_lambda: GAE λ parameter.

        Returns:
            (advantages, returns) — both shape (ptr,).
        """
        n = self.ptr
        advantages = torch.zeros(n, device=self.device)
        last_gae = 0.0

        for t in reversed(range(n)):
            if t == n - 1:
                nonterminal = 1.0 - float(next_done)
                nv = float(next_value)
            else:
                nonterminal = 1.0 - self.dones[t + 1].item()
                nv = self.values[t + 1].item()

            delta = (
                self.rewards[t].item()
                + gamma * nv * nonterminal
                - self.values[t].item()
            )
            last_gae = delta + gamma * gae_lambda * nonterminal * last_gae
            advantages[t] = last_gae

        returns = advantages + self.values[:n]
        return advantages, returns


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class PPOAgent(nn.Module):
    """PPO agent: Gaussian actor + scalar critic."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        device: torch.device,
        hidden: tuple[int, ...] = (64, 64),
    ) -> None:
        super().__init__()
        self.policy = PolicyNetwork(obs_dim, action_dim, hidden)
        self.value  = ValueNetwork(obs_dim, hidden)
        self.device = device
        self.to(device)

    @torch.no_grad()
    def act(
        self, obs: torch.Tensor, deterministic: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample an action (no gradient tracking).

        Args:
            obs:           Observation tensor, shape (obs_dim,) or (batch, obs_dim).
            deterministic: If True, return the distribution mean (eval mode).

        Returns:
            (action, logprob, value) — all tensors, no grad.
        """
        action, logprob = self.policy.act(obs, deterministic=deterministic)
        value = self.value(obs)
        return action, logprob, value

    def update(
        self,
        batch: dict[str, torch.Tensor],
        optimizer: torch.optim.Optimizer,
        clip_coef: float,
        vf_coef: float,
        ent_coef: float,
        max_grad_norm: float,
    ) -> dict[str, float]:
        """One minibatch PPO gradient step.

        Args:
            batch:         Dict with keys obs, actions, logprobs, advantages, returns.
            optimizer:     Shared Adam optimiser (caller owns it).
            clip_coef:     PPO ε clip coefficient.
            vf_coef:       Value loss weight.
            ent_coef:      Entropy bonus weight (0 = disabled).
            max_grad_norm: Gradient clip norm.

        Returns:
            Dict of scalar losses for logging.
        """
        obs        = batch["obs"]
        actions    = batch["actions"]
        old_lp     = batch["logprobs"]
        advantages = batch["advantages"]
        returns    = batch["returns"]

        new_lp, entropy = self.policy.evaluate(obs, actions)
        new_values      = self.value(obs)

        logratio = new_lp - old_lp
        ratio    = logratio.exp()

        # Clipped policy loss (PPO-Clip objective).
        pg1 = -advantages * ratio
        pg2 = -advantages * ratio.clamp(1.0 - clip_coef, 1.0 + clip_coef)
        policy_loss = torch.max(pg1, pg2).mean()

        # MSE value loss.
        value_loss = F.mse_loss(new_values, returns)

        # Entropy bonus (maximise → subtract from loss).
        entropy_loss = entropy.mean()

        loss = policy_loss + vf_coef * value_loss - ent_coef * entropy_loss

        # First-order approx KL for monitoring (no grad needed).
        with torch.no_grad():
            approx_kl = ((ratio - 1.0) - logratio).mean().item()

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.parameters(), max_grad_norm)
        optimizer.step()

        return {
            "policy_loss": policy_loss.item(),
            "value_loss":  value_loss.item(),
            "entropy":     entropy_loss.item(),
            "approx_kl":   approx_kl,
        }
