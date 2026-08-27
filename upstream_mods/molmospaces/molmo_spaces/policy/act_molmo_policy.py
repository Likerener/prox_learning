import sys
from typing import Any

import cv2
import numpy as np
import torch

from molmo_spaces.policy.base_policy import InferencePolicy, NONE_PHASE

ACT_ROOT = "/home/qinzhengfangli/act"
DETR_ROOT = "/home/qinzhengfangli/act/detr"

if ACT_ROOT not in sys.path:
    sys.path.insert(0, ACT_ROOT)

if DETR_ROOT not in sys.path:
    sys.path.insert(0, DETR_ROOT)

from imitate_episodes import make_policy


class ACTMolmoPolicy(InferencePolicy):
    def __init__(self, exp_config, task_type):
        super().__init__(exp_config, task_type)

        self.checkpoint_path = getattr(
            exp_config.policy_config,
            "checkpoint_path",
            "/home/qinzhengfangli/act/ckpt_molmo_chunk20_bs64_100ep_gripperfix/policy_best.ckpt",
        )

        self.chunk_size = 20
        self.state_dim = 9
        self.image_size = 64
        self.camera_name = "randomized_zed2_analogue_1"

        self.target_poses = {"grasp": np.eye(4)}
        self.current_phase = NONE_PHASE

        self.policy = None

        # ACT temporal aggregation buffer.
        self.t = 0
        self.all_time_actions = None
        self.temporal_agg = True
        self.temporal_agg_k = 0.01

        self.prepare_model()

    def prepare_model(self, model_name: str = ""):
        policy_config = {
            "lr": 1e-5,
            "num_queries": self.chunk_size,
            "kl_weight": 10,
            "hidden_dim": 512,
            "dim_feedforward": 3200,
            "lr_backbone": 1e-5,
            "backbone": "resnet18",
            "enc_layers": 4,
            "dec_layers": 7,
            "nheads": 8,
            "camera_names": ["top"],
            "state_dim": self.state_dim,
        }

        print(f"[ACTMolmoPolicy] loading checkpoint: {self.checkpoint_path}")
        self.policy = make_policy("ACT", policy_config)

        state_dict = torch.load(self.checkpoint_path, map_location="cuda")
        self.policy.load_state_dict(state_dict)

        self.policy.cuda()
        self.policy.eval()
        print("[ACTMolmoPolicy] checkpoint loaded")

    def reset(self):
        self.target_poses = {"grasp": np.eye(4)}
        self.current_phase = NONE_PHASE
        self.t = 0
        self.all_time_actions = None

    def obs_to_model_input(self, obs):
        return obs

    def inference_model(self, model_input):
        return model_input

    def model_output_to_action(self, model_output):
        return model_output

    def _qpos_to_tensor(self, obs: dict[str, Any]):
        qpos = obs["qpos"]

        arm = list(qpos.get("arm", []))
        gripper = list(qpos.get("gripper", []))

        vec = arm + gripper

        if len(vec) < self.state_dim:
            vec = vec + [0.0] * (self.state_dim - len(vec))

        vec = np.asarray(vec[:self.state_dim], dtype=np.float32)
        return torch.tensor(vec, dtype=torch.float32).cuda().unsqueeze(0)

    def _image_to_tensor(self, obs: dict[str, Any]):
        img = obs[self.camera_name]
        img = np.asarray(img)

        if img.shape[-1] == 4:
            img = img[..., :3]

        img = cv2.resize(
            img,
            (self.image_size, self.image_size),
            interpolation=cv2.INTER_AREA,
        )

        img = img.astype(np.float32)

        if img.max() > 1.0:
            img = img / 255.0

        img = np.transpose(img, (2, 0, 1))  # C,H,W
        img = img[None, None, ...]          # B,Ncam,C,H,W

        return torch.tensor(img, dtype=torch.float32).cuda()

    def get_action(self, observation: dict[str, Any]) -> dict[str, np.ndarray]:
        obs = observation[0] if isinstance(observation, list) else observation

        qpos_t = self._qpos_to_tensor(obs)
        image_t = self._image_to_tensor(obs)

        with torch.inference_mode():
            all_actions = self.policy(qpos_t, image_t)

        # ACT output shape: [B, chunk_size, action_dim]
        all_actions = all_actions.detach().cpu().numpy()[0].astype(np.float32)

        if self.temporal_agg:
            if self.all_time_actions is None:
                max_steps = 2000
                action_dim = all_actions.shape[-1]
                self.all_time_actions = np.zeros(
                    (max_steps, max_steps + self.chunk_size, action_dim),
                    dtype=np.float32,
                )

            self.all_time_actions[self.t, self.t:self.t + self.chunk_size, :] = all_actions

            actions_for_curr_step = self.all_time_actions[:self.t + 1, self.t, :]
            populated = np.any(actions_for_curr_step != 0, axis=1)
            actions_for_curr_step = actions_for_curr_step[populated]

            if len(actions_for_curr_step) == 0:
                action = all_actions[0]
            else:
                weights = np.exp(-self.temporal_agg_k * np.arange(len(actions_for_curr_step))[::-1])
                weights = weights / weights.sum()
                action = (actions_for_curr_step * weights[:, None]).sum(axis=0)
        else:
            action = all_actions[0]

        self.t += 1

        arm = action[:7].astype(np.float32)

        # MolmoSpaces Franka gripper action expects 1 dim, not 2 dims.
        gripper = np.asarray([action[7]], dtype=np.float32)

        return {
            "arm": arm,
            "gripper": gripper,
        }
