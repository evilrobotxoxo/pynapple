# IRIG-H Synchronization for Pynapple

## Context

Pynapple timestamps are relative to an undefined origin. To synchronize data streams across hardware, this feature adds IRIG-H timecode decoding, clock drift correction, and a `time_origin` attribute that anchors pynapple objects to UTC. No existing code breaks -- when `time_origin` is `None` (default), all behavior is identical.

## API Summary

### Synchronization -- anchor an object to an external time reference

| Function / Method | Purpose |
|---|---|
| `nap.irig_sync(data, pulses)` | Decode IRIG-H, correct drift, set `time_origin` |
| `.set_time_origin(origin)` | Return a new object with `time_origin` set (timestamps unchanged) |
| `nap.detect_ttl_pulses(signal)` | Extract pulse intervals from raw TTL waveform |
| `nap.decode_irig(pulses)` | Decode IRIG-H pulses into UTC timestamps (low-level) |

### Alignment -- shift a synchronized object to match another synchronized object

| Function / Method | Purpose |
|---|---|
| `.align_to(other)` | Return a new object with timestamps shifted to match `other`'s time origin |

### Concatenation

| Function / Method | Purpose |
|---|---|
| `nap.concatenate(*objects)` | Concatenate time series along the time axis with `time_origin` checking and propagation |

### Inspection

| Function / Method | Purpose |
|---|---|
| `.origin_datetime()` | UTC datetime of recording start (`time_origin` as datetime) |
| `.time_origin` | Float (unix timestamp of t=0) or `None` |

## Design

- **Internal storage:** drift-corrected relative timestamps (not unix time). Avoids breaking `restrict()`, `get()`, and manual `IntervalSet` creation.
- **`time_origin`:** optional attribute on all objects. `None` = not synced (default).
- **Compatibility enforcement:** operations between two objects error if one is synced and the other isn't, or if their `time_origin` values differ. Both `None` = no checking (backward compatible).
- **Synchronization** (anchoring an object to an external time reference): `irig_sync()` decodes IRIG + corrects drift + sets origin. `.set_time_origin()` stamps an origin on an object without changing timestamps (for objects that share a clock with an already-synced object).
- **Alignment** (shifting one synchronized object to match another): `.align_to(other)` shifts timestamps by the offset between two origins so both objects share a common timebase. Both objects must already be synchronized (have a `time_origin`) before alignment can occur.
- **Propagation:** `time_origin` propagates through `restrict()`, `count()`, `copy()`, `save()`/`load()`, etc.

## Implementation

### Step 1: `time_origin` on core classes

Files: `base_class.py`, `interval_set.py`, `ts_group.py`

- Add `time_origin=None` parameter to `__init__` of `_Base`, `IntervalSet`, `TsGroup`
- Store before `_initialized = True`
- Add `.origin_datetime()` method
- Add `.set_time_origin(origin)` method -- returns a new object with `time_origin` set, timestamps unchanged
- Add `.align_to(other)` method -- returns a new object with timestamps shifted to match `other`'s time origin (requires both objects to have `time_origin` set)
- `time_support` IntervalSets inherit `time_origin` from their parent

### Step 2: Compatibility checks

File: `base_class.py` (helper), inserted into all two-object operations

- `_check_time_origin_compatibility(obj_a, obj_b)` -- raises `TypeError` if mixing synced/unsynced, `ValueError` if origins differ
- Time series methods: `restrict()`, `value_from()`, `count(ep=...)`, `in_interval()`
- IntervalSet set operations: `intersect()`, `union()`, `set_diff()`, `in_interval()`
- TsGroup methods: `restrict()`, `value_from()`, `count(ep=...)`, `merge()`/`merge_group()`
- Concatenation: `nap.concatenate()` and the `np.concatenate` hook (`_concatenate_tsd`)

### Step 3: Propagation

Files: `time_series.py`, `interval_set.py`, `ts_group.py`, `utils.py`

- `_define_instance()` passes `self.time_origin` to new objects
- `_initialize_tsd_output()` propagates from input
- IntervalSet set operations propagate `time_origin`
- `_concatenate_tsd()` checks compatibility across inputs and propagates to output
- `_split_tsd()` propagates from input to each split piece

### Step 3b: `nap.concatenate()`

File: `pynapple/core/utils.py` or new file

- Dedicated `nap.concatenate(*objects)` function for concatenating time series along the time axis
- Checks `time_origin` compatibility across all inputs
- Requires strictly increasing, non-overlapping timestamps
- Propagates `time_origin` to the result
- Existing `np.concatenate` hook (`_concatenate_tsd`) also gains the same checks

### Step 4: IRIG module

File: `pynapple/core/irig.py` (new)

- `detect_ttl_pulses(signal, threshold)` -- Tsd of TTL values -> IntervalSet of pulses
- `decode_irig(pulses)` -- decode IRIG-H pulse train -> frames, mapping, quality
- `irig_sync(data, irig_pulses)` -- decode IRIG -> linear regression -> drift-corrected object + SyncInfo

### Step 5: Save/load

File: `pynapple/io/interface_npz.py`

- Save `time_origin` as NPZ key when not `None`; load and pass to constructor if present; old files load with `time_origin=None`

### Step 6: Display

- Add sync indicator to `__repr__` when `time_origin` is set

### Step 7: Exports

File: `pynapple/core/__init__.py`

- Export `irig_sync`, `detect_ttl_pulses`, `decode_irig`, `concatenate`

### Step 8: Tests

File: `tests/test_irig.py` (new) | Data: `tests/test_irig_data/` (gitignored)

- Helpers: `generate_irig_pulses()`, `generate_synced_recording()`
- Tests: IRIG decoding, drift correction, end-to-end sync, `time_origin` propagation, backward compatibility, compatibility errors, `.set_time_origin()`, `.align_to()`, input formats, edge cases

## Files

| File | Action |
|---|---|
| `pynapple/core/irig.py` | New |
| `pynapple/core/base_class.py` | Modify -- `time_origin`, `.origin_datetime()`, checks |
| `pynapple/core/time_series.py` | Modify -- propagation |
| `pynapple/core/interval_set.py` | Modify -- `time_origin`, propagation |
| `pynapple/core/ts_group.py` | Modify -- `time_origin`, propagation, checks |
| `pynapple/core/utils.py` | Modify -- `nap.concatenate()`, `time_origin` in `_concatenate_tsd`/`_split_tsd` |
| `pynapple/core/__init__.py` | Modify -- exports |
| `pynapple/io/interface_npz.py` | Modify -- save/load |
| `tests/test_irig.py` | New |
| `.gitignore` | Modify -- add `tests/test_irig_data/` |

## Verification

1. `pytest tests/` -- all existing tests pass
2. `pytest tests/test_irig.py -v` -- new tests pass
3. Lint passes
