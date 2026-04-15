"""
LeRobot pi0 LoRA policy adapter for the airoa evaluation framework.

Implements BasePolicy.infer() for the WebSocket server.
Input:  head_rgb (H,W,3), hand_rgb (H,W,3), state (8,), prompt (str)
Output: actions (T, 11)

Checkpoint directory layout expected:
    checkpoint_dir/
    ├── adapter_config.json           # LoRA adapter config
    ├── adapter_model.safetensors     # LoRA weights
    ├── base_model/                   # pi0 base model + tokenizer
    │   ├── config.json
    │   ├── model.safetensors
    │   ├── tokenizer.json
    │   └── tokenizer_config.json
    └── stats.json                    # normalization stats
"""

import json
import logging
from pathlib import Path

import numpy as np
import torch
from policy_client import base_policy

logger = logging.getLogger(__name__)


def normalize_state(state: np.ndarray, mean: list, std: list) -> np.ndarray:
    mean = np.array(mean, dtype=np.float32)
    std = np.array(std, dtype=np.float32)
    return (state - mean) / np.maximum(std, 1e-8)


def unnormalize_action(action: np.ndarray, mean: list, std: list) -> np.ndarray:
    mean = np.array(mean, dtype=np.float32)
    std = np.array(std, dtype=np.float32)
    return action * std + mean


class LeRobotPi0Policy(base_policy.BasePolicy):
    """Wraps a LeRobot pi0 + LoRA model behind the BasePolicy interface."""

    def __init__(self, checkpoint_dir: str, device: str = "cuda"):
        self.device = torch.device(device)

        ckpt_path = Path(checkpoint_dir)
        base_model_dir = str(ckpt_path / "base_model")
        stats_path = str(ckpt_path / "stats.json")
        adapter_dir = str(ckpt_path)

        for p, name in [(base_model_dir, "base_model/"), (stats_path, "stats.json")]:
            if not Path(p).exists():
                raise FileNotFoundError(f"Required {name} not found in checkpoint: {p}")

        logger.info("Loading dataset stats from %s", stats_path)
        with open(stats_path) as f:
            all_stats = json.load(f)

        self.state_mean = all_stats["observation.state"]["mean"]
        self.state_std = all_stats["observation.state"]["std"]
        self.action_mean = all_stats["action.relative"]["mean"]
        self.action_std = all_stats["action.relative"]["std"]

        logger.info("Loading base pi0 model from %s", base_model_dir)

        import lerobot.datasets.utils as du
        _orig = du.get_safe_version
        def _patched(repo_id, version=None):
            if "/" not in repo_id or repo_id.startswith("local"):
                return f"v{version}" if version and not str(version).startswith("v") else str(version or "main")
            return _orig(repo_id, version)
        du.get_safe_version = _patched

        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies.factory import make_policy

        policy_cfg = PreTrainedConfig.from_pretrained(base_model_dir)
        policy_cfg.pretrained_path = Path(base_model_dir)

        ds_meta = self._make_ds_meta(all_stats)

        rename_map = {
            "observation.image.head": "observation.images.head",
            "observation.image.hand": "observation.images.hand",
        }
        self.policy = make_policy(cfg=policy_cfg, ds_meta=ds_meta, rename_map=rename_map)

        logger.info("Loading LoRA adapter from %s", adapter_dir)
        from peft import PeftModel
        self.policy = PeftModel.from_pretrained(self.policy, adapter_dir)

        self.policy = self.policy.to(self.device)
        self.policy.eval()
        logger.info("Model loaded on %s", self.device)

        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_dir)

        self.chunk_size = policy_cfg.chunk_size
        self.max_state_dim = policy_cfg.max_state_dim
        self.max_action_dim = policy_cfg.max_action_dim
        self.num_inference_steps = policy_cfg.num_inference_steps
        self.tokenizer_max_length = policy_cfg.tokenizer_max_length

        logger.info(
            "Config: chunk_size=%d, max_state_dim=%d, max_action_dim=%d, inference_steps=%d",
            self.chunk_size, self.max_state_dim, self.max_action_dim, self.num_inference_steps,
        )

    def _make_ds_meta(self, all_stats):
        class MockMeta:
            def __init__(self, stats):
                self.stats = stats
                self.features = {
                    "observation.state": {
                        "dtype": "float32",
                        "shape": [8],
                        "names": ["arm_lift_joint", "arm_flex_joint", "arm_roll_joint",
                                  "wrist_flex_joint", "wrist_roll_joint", "hand_motor_joint",
                                  "head_pan_joint", "head_tilt_joint"],
                    },
                    "observation.images.head": {
                        "dtype": "video",
                        "shape": [480, 640, 3],
                        "names": ["height", "width", "channel"],
                    },
                    "observation.images.hand": {
                        "dtype": "video",
                        "shape": [480, 640, 3],
                        "names": ["height", "width", "channel"],
                    },
                    "action": {
                        "dtype": "float32",
                        "shape": [11],
                        "names": ["arm_lift_joint", "arm_flex_joint", "arm_roll_joint",
                                  "wrist_flex_joint", "wrist_roll_joint", "hand_motor_joint",
                                  "head_pan_joint", "head_tilt_joint", "base_x", "base_y", "base_t"],
                    },
                }

        needed_stats = {}
        for key in ["observation.state", "action"]:
            src = all_stats["action.relative"] if key == "action" else all_stats[key]
            needed_stats[key] = {
                "mean": torch.tensor(src["mean"], dtype=torch.float32),
                "std": torch.tensor(src["std"], dtype=torch.float32),
            }
        return MockMeta(needed_stats)

    @torch.no_grad()
    def infer(self, obs: dict) -> dict:
        head_rgb = np.array(obs["head_rgb"], dtype=np.uint8, copy=True)
        hand_rgb = np.array(obs["hand_rgb"], dtype=np.uint8, copy=True)

        head_tensor = torch.from_numpy(head_rgb).permute(2, 0, 1).float() / 255.0
        hand_tensor = torch.from_numpy(hand_rgb).permute(2, 0, 1).float() / 255.0
        head_tensor = head_tensor.unsqueeze(0).to(self.device)
        hand_tensor = hand_tensor.unsqueeze(0).to(self.device)

        state = np.asarray(obs["state"], dtype=np.float32)
        state_norm = normalize_state(state, self.state_mean, self.state_std)
        state_tensor = torch.from_numpy(state_norm).unsqueeze(0).to(self.device)

        prompt = obs.get("prompt") or "do something"
        prompt = prompt + "\n"
        tokens = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer_max_length,
            truncation=True,
            return_tensors="pt",
        )
        lang_tokens = tokens["input_ids"].to(self.device)
        lang_mask = tokens["attention_mask"].bool().to(self.device)

        batch = {
            "observation.images.head": head_tensor,
            "observation.images.hand": hand_tensor,
            "observation.state": state_tensor,
            "observation.language.tokens": lang_tokens,
            "observation.language.attention_mask": lang_mask,
        }

        self.policy.reset()
        actions = self.policy.predict_action_chunk(batch)

        actions = actions[0].cpu().numpy()
        actions = actions[:, :11]
        actions = unnormalize_action(actions, self.action_mean, self.action_std)
        actions = np.nan_to_num(actions, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

        logger.info("Inference done: actions shape=%s, range=[%.4f, %.4f]",
                     actions.shape, actions.min(), actions.max())

        return {"actions": actions}

    def reset(self) -> None:
        if hasattr(self.policy, "reset"):
            self.policy.reset()

    @property
    def metadata(self) -> dict:
        return {
            "model": "lerobot-pi0-lora",
            "chunk_size": self.chunk_size,
            "action_dim": 11,
            "state_dim": 8,
        }
