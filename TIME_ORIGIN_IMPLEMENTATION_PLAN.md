# Plan: Implement `time_origin` Infrastructure in Pynapple

## Context

Pynapple timestamps are relative to an undefined origin. To support multi-stream synchronization, we're adding a `time_origin` attribute to all core objects. This anchors timestamps to UTC so pynapple can enforce consistency when combining objects from different recording systems. IRIG-specific decoding lives in a separate repo ([neurokairos-pynapple](https://github.com/SjulsonLab/neurokairos-pynapple)).

Branch: `irig_test`

### Already done
- `_check_time_origin_compatibility()` helper in `base_class.py:18`
- `time_origin` param on `_Base.__init__` (`base_class.py:65`), stored at line 66
- `origin_datetime()` on `_Base` (`base_class.py:96`)
- Compatibility check in `value_from()` (`base_class.py`)

### Not yet done
- Constructors of `_BaseTsd`, `Tsd`, `TsdFrame`, `TsdTensor`, `Ts` do NOT pass `time_origin` through to `_Base`
- Everything else below

---

## Step 1: Thread `time_origin` through time series constructors

**File: `pynapple/core/time_series.py`**

| Constructor | Line | Change |
|---|---|---|
| `_BaseTsd.__init__` | 174 | Add `time_origin=None` param, pass to `super().__init__(..., time_origin=time_origin)` |
| `TsdTensor.__init__` | ~954 | Add `time_origin=None` param, pass to `super().__init__` |
| `TsdFrame.__init__` | ~1526 | Add `time_origin=None` param, pass to `super().__init__` |
| `Tsd.__init__` | ~2717 | Add `time_origin=None` param, pass to `super().__init__` |
| `Ts.__init__` | 3343 | Add `time_origin=None` param, pass to `super().__init__(..., time_origin=time_origin)` |

---

## Step 2: Add `time_origin` to IntervalSet

**File: `pynapple/core/interval_set.py`**

- Add `time_origin=None` param to `IntervalSet.__init__` (line ~181)
- Store `self.time_origin = time_origin` before `_initialized = True` (line ~300)
- Add `origin_datetime()` method (same as `_Base` version)
- Add `set_time_origin(origin)` — returns new IntervalSet with same data + origin set
- Add `sync_to(other)` — shifts start/end by offset, sets `time_origin` to `other.time_origin`
- Import `_check_time_origin_compatibility` from `base_class`

---

## Step 3: Add `time_origin` to TsGroup

**File: `pynapple/core/ts_group.py`**

- Add `time_origin=None` param to `TsGroup.__init__` (line ~193)
- Store `self.time_origin = time_origin` before `_initialized = True` (line ~310)
- Add `origin_datetime()`, `set_time_origin(origin)`, `sync_to(other)` methods
- `sync_to` must shift every contained Ts/Tsd object's timestamps + the time_support
- Import `_check_time_origin_compatibility` from `base_class`

---

## Step 4: Propagation

### 4a: `_initialize_tsd_output()` (`time_series.py:74`)

At line ~160, before `return cls(...)`:
```python
to = getattr(input_object, "time_origin", None)
if to is not None:
    kwargs.setdefault("time_origin", to)
```

### 4b: `Ts._define_instance()` (`time_series.py:3370`)

The `values is None` branch (line 3377) creates `Ts` directly — pass `time_origin=self.time_origin`. The other branch goes through `_initialize_tsd_output` (handled by 4a).

### 4c: IntervalSet operations (`interval_set.py`)

Pass `time_origin=self.time_origin` to output IntervalSet in:
- `intersect()`, `union()`, `set_diff()`, `time_span()`, `merge_close_intervals()`, `split()`
- `__getitem__` (all branches that return IntervalSet)
- `drop_short_intervals()`, `drop_long_intervals()`

### 4d: TsGroup operations (`ts_group.py`)

Pass `time_origin=self.time_origin` in:
- `_ts_group_from_keys()` (used by `__getitem__`)
- `restrict()`, `value_from()`, `get()`, `merge_group()`

### 4e: `_concatenate_tsd()` (`utils.py:248`)

Collect `time_origin` from all inputs, check compatibility, pass shared origin to output constructor. `_split_tsd()` uses `_define_instance()` so propagates automatically.

---

## Step 5: Add `set_time_origin()` and `sync_to()` to `_Base`

**File: `pynapple/core/base_class.py`**

- `set_time_origin(origin)`: Reconstruct object via constructor with `time_origin=origin`, timestamps unchanged. Use `object.__setattr__` to bypass immutability if needed.
- `sync_to(other)`: Compute `offset = self.time_origin - other.time_origin`. Shift timestamps by offset, shift time_support by offset. Set `time_origin = other.time_origin`. Errors if either object lacks `time_origin`.

---

## Step 6: Compatibility checks in remaining operations

**`base_class.py`** — insert `_check_time_origin_compatibility(self, ep/iset)` into:
- `restrict()` (~line 408)
- `count(ep=...)` (~line 321, skip if `ep is self.time_support`)
- `in_interval()` (~line 433)

**`interval_set.py`** — insert checks into:
- `intersect()`, `union()`, `set_diff()`, `in_interval()`

**`ts_group.py`** — insert checks into:
- `restrict()`, `value_from()`, `count(ep=...)`, `merge_group()`

---

## Step 7: `nap.concatenate()`

**File: `pynapple/core/utils.py`** (new public function)

```python
def concatenate(*objects):
```

- **All types**: Check `time_origin` compatibility across all inputs. Propagate to result.
- **Time series** (Tsd/TsdFrame/TsdTensor/Ts): Concatenate timestamps + data. Auto-sort with warning if unsorted. Union time supports.
- **IntervalSet**: Concatenate start/end arrays, construct new IntervalSet (equivalent to multi-arg union).
- **TsGroup**: Union with merge — overlapping keys get their Ts/Tsd objects concatenated. Non-overlapping keys kept as-is. Metadata: keep first group's values for overlapping keys.

**Export** from `pynapple/core/__init__.py` and `pynapple/__init__.py`.

---

## Step 8: Save/load

**Save** — add `time_origin` to `np.savez()` when not None:
- `Tsd.save()`, `TsdFrame.save()`, `TsdTensor.save()`, `Ts.save()` in `time_series.py`
- `IntervalSet.save()` in `interval_set.py`
- `TsGroup.save()` in `ts_group.py`

**Load** — in `_from_npz_reader` methods:
- `base_class.py` (~line 710): Extract `time_origin` from file, pass to constructor
- `interval_set.py` (~line 594): Same
- `ts_group.py` (~line 1634): Same
- Old files without key → `time_origin=None` (backward compatible)

**File: `pynapple/io/interface_npz.py`** — pass `time_origin` through if present

---

## Step 9: `__repr__` sync indicator

When `time_origin is not None`, append to repr output:
```
time_origin: 2024-01-15 14:30:00 UTC
```

Files: `Tsd.__repr__`, `TsdFrame.__repr__`, `TsdTensor.__repr__`, `Ts.__repr__` in `time_series.py`; `IntervalSet.__repr__` in `interval_set.py`; `TsGroup.__repr__` in `ts_group.py`

---

## Step 10: Tests

**File: `tests/test_time_origin.py`** (new)

1. Construction with `time_origin` — all types accept it, default is None
2. `origin_datetime()` returns correct UTC datetime
3. Backward compatibility — all ops work with `time_origin=None`
4. Compatibility checks — TypeError for synced+unsynced, ValueError for different origins
5. Propagation — `restrict()`, `count()`, numpy ops, IntervalSet ops, TsGroup slicing
6. `set_time_origin()` — returns new object, original unchanged, timestamps unchanged
7. `sync_to()` — correct timestamp shift, result has other's origin, errors when no origin
8. `nap.concatenate()` — all types, propagation, sort+warn, compatibility errors, TsGroup key merge
9. Save/load round-trip — all types, old files without time_origin
10. `__repr__` — shows origin when set, absent when None

---

## Key gotchas

1. **Immutability**: `set_time_origin`/`sync_to` return new objects. Use `object.__setattr__` or reconstruct via constructor.
2. **time_support IntervalSets**: Internal time_support should NOT independently carry `time_origin` — it's part of the parent object.
3. **`_initialize_tsd_output` kwargs**: Use `kwargs.setdefault()` not `kwargs[]=` to avoid overwriting caller-provided values.
4. **TsGroup `sync_to`**: Must shift every contained Ts/Tsd + time_support. More involved than other types.
5. **`nap.concatenate` vs `np.concatenate` hook**: `nap.concatenate` auto-sorts with warning; the existing `_concatenate_tsd` hook stays strict (errors on unsorted).

---

## Verification

1. `pytest tests/` — all existing tests still pass
2. `pytest tests/test_time_origin.py -v` — new tests pass
3. `black --check pynapple tests && isort --check pynapple tests --profile black && flake8 pynapple --max-complexity 10`
