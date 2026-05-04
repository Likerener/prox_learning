"""Data layer.

Top-level exports:
    PLADataset, collate_pla    — sliding-window training DataLoader
    compute_stats, normalize, unnormalize  — per-channel stats
    verify_dataset             — post-collection deep audit
    validate, proximity_informative_fraction  — schema utilities
    collect_episode            — single-episode HDF5 writer
    audit_episode, collector_should_stop  — streaming sentinel API
"""
from pla.data.collect import collect_episode
from pla.data.dataset import PLADataset, collate_pla
from pla.data.normalize import compute_stats, normalize, unnormalize
from pla.data.schema import proximity_informative_fraction, validate
from pla.data.sentinel import audit_episode, collector_should_stop
from pla.data.verify import verify_dataset

__all__ = [
    "PLADataset",
    "audit_episode",
    "collate_pla",
    "collect_episode",
    "collector_should_stop",
    "compute_stats",
    "normalize",
    "proximity_informative_fraction",
    "unnormalize",
    "validate",
    "verify_dataset",
]
