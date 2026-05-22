"""ACT + proximity-encoder integration (P+ACT).

Modules:
  - build_mapping: build prox_mapping.json that links act_style_data episodes to
    their source h5 trajectories by exact-qpos match.
  - dataset: ProxAugmentedEpisodicDataset, wraps ACT's EpisodicDataset and
    returns a per-sensor temporal proximity window alongside the usual tuple.
  - prox_features: FrozenProxFeatureExtractor, the frozen 0.82M-param transformer
    encoder turned into a (B, N_sensors, 3) feature module for ACT to consume.
  - imitate_episodes_with_prox: ACT trainer with --use_proximity (forked from
    submodules/act/imitate_episodes.py).
  - eval_act_with_prox_encoder: rollout-time eval with a live proximity buffer.
"""
