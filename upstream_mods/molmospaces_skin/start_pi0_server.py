import logging
from pathlib import Path

import numpy as np

from molmo_spaces.evaluation.policy_server import WebsocketPolicyServer

logging.basicConfig(level=logging.INFO)


def decode_ndarray_dict(x):
    """Decode websocket-serialized numpy arrays back to real np.ndarray."""
    if isinstance(x, dict):
        keys = set(x.keys())

        # msgpack may preserve bytes keys
        if b"__ndarray__" in keys:
            arr = np.frombuffer(x[b"data"], dtype=np.dtype(x[b"dtype"]))
            return arr.reshape(x[b"shape"])

        # or string keys
        if "__ndarray__" in keys:
            arr = np.frombuffer(x["data"], dtype=np.dtype(x["dtype"]))
            return arr.reshape(x["shape"])

        return {k: decode_ndarray_dict(v) for k, v in x.items()}

    if isinstance(x, list):
        return [decode_ndarray_dict(v) for v in x]

    return x


def fix_1d(x, dtype=np.float32):
    arr = np.asarray(x, dtype=dtype)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    return arr.reshape(-1)


class OpenPIServerPolicy:
    def __init__(self, checkpoint_path: str):
        self.checkpoint_path = checkpoint_path
        self.model = None

    def prepare_model(self):
        from openpi.policies import policy_config as _policy_config
        from openpi.training import config as _config

        config_name = Path(self.checkpoint_path.rstrip("/")).name
        logging.info(f"Loading OpenPI config: {config_name}")
        train_config = _config.get_config(config_name)

        logging.info(f"Loading OpenPI checkpoint: {self.checkpoint_path}")
        self.model = _policy_config.create_trained_policy(train_config, self.checkpoint_path)

        logging.info("OpenPI model loaded.")

    def obs_to_model_input(self, obs):
        return obs

    def inference_model(self, model_input):
        model_input = decode_ndarray_dict(model_input)

        model_input["observation/joint_position"] = fix_1d(
            model_input["observation/joint_position"]
        )
        model_input["observation/gripper_position"] = fix_1d(
            model_input["observation/gripper_position"]
        )

        return self.model.infer(model_input)

    def model_output_to_action(self, model_output):
        # WebsocketPolicyServer expects a dict because it adds action["server_timing"].
        actions = model_output["actions"]

        # Avoid accidental nesting.
        if isinstance(actions, dict):
            if "actions" in actions:
                actions = actions["actions"]
            elif b"actions" in actions:
                actions = actions[b"actions"]

        # Important: convert numpy array to plain list before websocket sends it back.
        # Otherwise the eval client receives a serialized ndarray dict and actions_buffer[0] fails.
        if hasattr(actions, "tolist"):
            actions = actions.tolist()

        return {"actions": actions}


policy = OpenPIServerPolicy(
    checkpoint_path="/home/qinzhengfangli/checkpoints/paligemma_diffusion_droid"
)

server = WebsocketPolicyServer(
    policies=policy,
    model_name="paligemma_diffusion_droid",
    host="localhost",
    port=8080,
)

server.serve_forever()
