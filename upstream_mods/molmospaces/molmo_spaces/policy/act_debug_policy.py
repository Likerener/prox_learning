import numpy as np
from typing import Any

from molmo_spaces.policy.base_policy import InferencePolicy, NONE_PHASE


class ACTDebugPolicy(InferencePolicy):
    printed = False

    def __init__(self, config, task_type):
        super().__init__(config, task_type)
        self.target_poses = {"grasp": np.eye(4)}
        self.current_phase = NONE_PHASE

    def prepare_model(self, model_name: str = ""):
        pass

    def obs_to_model_input(self, obs):
        return obs

    def inference_model(self, model_input):
        return model_input

    def model_output_to_action(self, model_output):
        return model_output

    def reset(self):
        self.target_poses = {"grasp": np.eye(4)}
        self.current_phase = NONE_PHASE

    def get_action(self, observation: dict[str, Any]) -> dict[str, Any]:
        obs = observation[0] if isinstance(observation, list) else observation

        if not ACTDebugPolicy.printed:
            ACTDebugPolicy.printed = True
            print("\n===== ACT DEBUG OBSERVATION =====")
            print("type:", type(obs))
            print("keys:", obs.keys() if isinstance(obs, dict) else None)

            if isinstance(obs, dict):
                for k, v in obs.items():
                    if isinstance(v, dict):
                        print(k, "DICT keys:", v.keys())
                        for kk, vv in list(v.items())[:80]:
                            print(" ", kk, type(vv), getattr(vv, "shape", None))
                    else:
                        print(k, type(v), getattr(v, "shape", None))
            print("===== END DEBUG OBSERVATION =====\n")

        # Hold default/current-ish pose.
        arm = np.array([0, -0.7853, 0, -2.35619, 0, 1.57079, 0.0], dtype=np.float32)
        gripper = np.array([0.00296, 0.00296], dtype=np.float32)

        try:
            if isinstance(obs, dict) and "robot_state" in obs and "qpos" in obs["robot_state"]:
                qpos = obs["robot_state"]["qpos"]
                if isinstance(qpos, dict):
                    arm = np.asarray(qpos.get("arm", arm), dtype=np.float32)
                    gripper = np.asarray(qpos.get("gripper", gripper), dtype=np.float32)
                else:
                    arr = np.asarray(qpos, dtype=np.float32).reshape(-1)
                    if len(arr) >= 7:
                        arm = arr[:7]
                    if len(arr) >= 9:
                        gripper = arr[7:9]
        except Exception as e:
            print("debug qpos parse failed:", repr(e))

        return {
            "arm": arm,
            "gripper": gripper,
        }
