"""Transformer encoder that maps raw proximity-sensor time series to
3D object position in the sensor frame.

Layout:
    cache.py    — h5 → flat npz shards (one sample = one window).
    dataset.py  — torch Dataset over the npz shards.
    model.py    — transformer encoder-only architecture.
"""
