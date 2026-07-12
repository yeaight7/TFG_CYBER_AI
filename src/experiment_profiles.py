"""Authoritative, versioned scientific experiment profiles."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any


def _canonical_json(content: dict[str, Any]) -> str:
    return json.dumps(content, sort_keys=True, separators=(",", ":"), allow_nan=False)


@dataclass(frozen=True)
class ExperimentProfile:
    """Immutable profile backed by canonical JSON for stable hashing and copies."""

    profile_id: str
    _canonical_content: str

    @classmethod
    def from_content(cls, profile_id: str, content: dict[str, Any]) -> ExperimentProfile:
        return cls(profile_id=profile_id, _canonical_content=_canonical_json(content))

    @property
    def content_hash(self) -> str:
        return hashlib.sha256(self._canonical_content.encode("utf-8")).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return json.loads(self._canonical_content)

    def qrdqn_hyperparams(self) -> dict[str, Any]:
        content = self.to_dict()
        content.pop("reward_config")
        return content

    def reward_config(self) -> dict[str, float]:
        return self.to_dict()["reward_config"]


_MAIN_V1_CONTENT: dict[str, Any] = {
    "policy": "MlpPolicy",
    "policy_kwargs": {
        "net_arch": [1024, 1024, 512],
        "n_quantiles": 200,
    },
    "learning_rate": 5e-5,
    "buffer_size": 1_000_000,
    "learning_starts": 50_000,
    "batch_size": 2_048,
    "gamma": 0.0,
    "tau": 1.0,
    "train_freq": 100,
    "gradient_steps": 20,
    "target_update_interval": 10_000,
    "exploration_initial_eps": 1.0,
    "exploration_final_eps": 0.02,
    "exploration_fraction": 0.10,
    "max_grad_norm": 10.0,
    "reward_config": {
        "tp": 1.5,
        "fp": -2.0,
        "fn": -5.0,
        "omission": 0.0,
    },
}

MAIN_V1_PROFILE = ExperimentProfile.from_content("main-v1", _MAIN_V1_CONTENT)
MAIN_V1_PROFILE_HASH = MAIN_V1_PROFILE.content_hash

_PROFILES = {MAIN_V1_PROFILE.profile_id: MAIN_V1_PROFILE}


def get_experiment_profile(profile_id: str) -> ExperimentProfile:
    try:
        return _PROFILES[profile_id]
    except KeyError as exc:
        raise ValueError(
            f"Unknown experiment profile '{profile_id}'. Available: {sorted(_PROFILES)}"
        ) from exc
