# Proposal: Time Origin Infrastructure and IRIG-H Synchronization for Pynapple

## The problem

Modern neuroscience experiments routinely combine data from multiple acquisition systems: electrophysiology, imaging, behavioral tracking, optogenetic stimulation, and so on. Each system records on its own clock. To analyze these streams together, researchers need to align them to a common timebase. Today, pynapple has no opinion about how timestamps relate to wall-clock time. Every object's time axis starts at some arbitrary zero, and there is no built-in way to express that two objects were recorded simultaneously or to detect when they were not.

In practice this means users align their data before loading it into pynapple, using ad-hoc scripts that vary from lab to lab. This is error-prone: silent clock drift accumulates over long recordings, off-by-one bugs in pulse counting go undetected, and there is no safety net if a user accidentally combines time series from different sessions.

## Multi-repo architecture

The implementation is split across three repositories:

- **pynapple** (this repo): General-purpose `time_origin` infrastructure -- the attribute, propagation, compatibility checks, `set_time_origin()`, `sync_to()`, and `nap.concatenate()`.
- **[neurokairos](https://github.com/SjulsonLab/neurokairos)**: Core IRIG-H decoding and clock mapping -- TTL signal processing (`auto_threshold()`, `detect_edges()`, `measure_pulse_widths()`), IRIG-H frame decoding (`decode_dat_irig()`, `decode_intervals_irig()`, `decode_sglx_irig()`, `decode_video_irig()`), and the `ClockTable` dataclass for sparse time mapping between local and UTC clocks. Pure numpy, no pynapple dependency.
- **[neurokairos-pynapple](https://github.com/SjulsonLab/neurokairos-pynapple)**: Thin pynapple integration layer. Provides `BinaryFile`, which wraps a memory-mapped binary data file + `ClockTable` to produce pynapple `Tsd`/`TsdFrame` on demand. Re-exports core neurokairos symbols for convenience.

The `time_origin` attribute is general-purpose and useful for any multi-stream synchronization workflow. IRIG decoding is one specific way to set `time_origin`, but labs may use other methods (PTP, GPS, manual timestamps). Keeping the IRIG code in separate packages keeps pynapple's core focused while still providing a reference implementation for IRIG users.

## What IRIG provides

IRIG (Inter-Range Instrumentation Group) timecodes are a family of standardized timing signals widely used in instrumentation and increasingly adopted in neuroscience hardware. An IRIG encoder broadcasts a continuous pulse train that encodes UTC time. A recording system captures this pulse train alongside its neural data. By decoding the pulses, we can determine the exact UTC time corresponding to any sample in the recording, and we can measure and correct clock drift between the local oscillator and the UTC reference.

This proposal focuses on IRIG-H (one pulse per minute, 60 BCD-encoded pulses per frame), which is the format used by several common neuroscience acquisition systems. The architecture is general enough to support other IRIG variants in the future.

## What this proposal adds

### Core layer (pynapple): `time_origin` attribute

Every pynapple object (Tsd, TsdFrame, TsdTensor, Ts, IntervalSet, TsGroup) gains an optional `time_origin` attribute. This is a single float64 representing the Unix timestamp of the object's t=0 (the number of seconds since midnight on Jan 1, 1970). When `time_origin` is `None` (the default), the object behaves exactly as it does today -- no existing code is affected. When it is set, pynapple knows that the object's timestamps are anchored to real-world time, and it can enforce consistency: operations that combine two objects will raise an error if their time origins differ or if one is synchronized and the other is not. This catches a class of bugs that currently pass silently.

A key design choice is that timestamps remain relative. We do not convert the time axis to Unix time. Instead, `time_origin` records the offset, and the timestamps stay as small, human-readable floats (seconds from recording start). This means `restrict()`, `get()`, and manual `IntervalSet` construction continue to work with the same kinds of values users are accustomed to. A convenience method, `.origin_datetime()`, converts the origin to a Python `datetime` for display and interoperability.

`time_origin` propagates automatically. When an operation produces a new object -- `restrict()`, `count()`, `copy()`, slicing, numpy operations -- the result inherits the origin of its input. This propagation follows the existing `_define_instance()` pattern, so it requires minimal changes to the codebase and does not introduce new control flow. Save and load round-trip the attribute through NPZ files, remaining backward-compatible with files that predate the feature.

### IRIG layer (neurokairos + neurokairos-pynapple): decoding and drift correction

The **neurokairos** package provides the core IRIG decoding pipeline:

`auto_threshold()` uses Otsu's method to find the TTL threshold in a raw signal. `detect_edges()` finds rising/falling edges, and `measure_pulse_widths()` computes pulse intervals. These are general utilities useful beyond IRIG for any TTL-based event detection.

`decode_dat_irig()` and `decode_sglx_irig()` are the main entry points for decoding IRIG-H from raw binary recordings (generic `.dat` files and SpikeGLX `.bin` files, respectively). `decode_intervals_irig()` decodes from pre-extracted pulse intervals. `decode_video_irig()` decodes from video files with a visible IRIG LED. All decoders return a `ClockTable` -- a sparse mapping between local sample times and UTC, with bidirectional interpolation and JSON-serializable metadata. `ClockTable` handles partial frames, corrupted pulses, and concatenated recordings gracefully, and supports save/load to NPZ files.

The **neurokairos-pynapple** package provides `BinaryFile`, which wraps a memory-mapped binary data file together with a `ClockTable` to produce pynapple `Tsd`/`TsdFrame` objects on demand. `BinaryFile` avoids materializing timestamps for entire large recordings, validates the constant-rate assumption against the `ClockTable`, and detects concatenation boundaries. It exposes `set_time_origin()` and `sync_to()` methods that mirror pynapple's API.

### Decoding and synchronization

Two instance methods on pynapple objects complete the workflow. They correspond to two distinct steps -- **decoding/time-referencing** and **synchronization** -- that are worth defining clearly:

**Decoding** (time-referencing) means extracting UTC timestamps from an IRIG signal and anchoring an object to that external time reference. It answers the question "when, in real-world time, did this recording start?" There are two ways to time-reference an object: the neurokairos decoders (`decode_dat_irig()`, `decode_sglx_irig()`, etc.) produce a `ClockTable` whose `time_origin` can be applied to pynapple objects via `.set_time_origin()`, while for objects that share a clock with an already time-referenced object, you can stamp the same origin directly. For example, if an electrophysiology system records both neural data and a behavior camera on the same clock, you decode the neural data with IRIG and then stamp the same origin onto the behavior data:

```python
import neurokairos as nk

clock_table = nk.decode_sglx_irig(bin_path)
ephys_synced = ephys.set_time_origin(clock_table.time_origin)
behavior_synced = behavior.set_time_origin(clock_table.time_origin)
```

**Synchronization** is the next step: shifting one time-referenced object's timestamps to match another time-referenced object's timebase. It answers the question "how do I put these two streams on the same clock?" Synchronization requires that both objects have already been time-referenced (i.e., both have a `time_origin`). Calling `A.sync_to(B)` returns a new object whose timestamps have been shifted by the offset between the two origins, with `time_origin` set to match B's. The directionality is explicit -- A is being adjusted, B is the reference -- and it composes naturally when synchronizing multiple streams:

```python
spikes_synced = spikes.sync_to(lfp)
stim_synced   = stim.sync_to(lfp)
```

Calling `.sync_to()` on an object that has no `time_origin` raises an error, because there is no offset to compute. This enforces the two-step model: decode first, then synchronize.

**Concatenation.** Currently, pynapple objects can be concatenated via `np.concatenate`, which pynapple intercepts through numpy's `__array_function__` protocol. With `time_origin`, this hook gains compatibility checking (all inputs must have the same origin or all be `None`) and propagation. A new dedicated function, `nap.concatenate()`, provides the same functionality with an explicit pynapple entry point.

**Compatibility checking across all two-object operations.** Beyond the methods described above, `time_origin` compatibility is enforced wherever two pynapple objects are combined: `restrict()`, `count()`, `in_interval()`, IntervalSet set operations (`intersect()`, `union()`, `set_diff()`), and TsGroup methods (`merge()`, `value_from()`). In every case, if one object is synchronized and the other is not, or if their origins differ, the operation raises an error. When both origins are `None`, no checking occurs and behavior is identical to today.

## Why this belongs in pynapple

Time referencing and synchronization are not niche concerns. They are prerequisites for most multi-modal analyses, and getting either step wrong invalidates results. By handling both at the container level, pynapple can prevent timing errors structurally rather than relying on users to be careful. The `time_origin` attribute is cheap (one float per object, no performance impact) and invisible when unused, so it imposes no burden on users who do not need it.

The IRIG decoding logic is more specialized and lives in the separate `neurokairos` package, with pynapple integration in `neurokairos-pynapple`. IRIG timecodes are an open standard used by hardware from multiple vendors, and providing a reference decoder saves every lab from writing their own. Keeping the decoding in separate packages means pynapple's core stays focused on general-purpose time series operations, while users who need IRIG can `pip install neurokairos neurokairos-pynapple`.

## Backward compatibility

Every change defaults to `None`. Existing code that does not pass `time_origin` sees no change in behavior. Existing NPZ files load without issues. The compatibility checks only activate when at least one object in an operation has a non-`None` origin. Pynapple gains new public functions (`concatenate`) and core classes gain new methods (`.set_time_origin()`, `.sync_to()`, `.origin_datetime()`), but no existing method signature or behavior is modified.

## Scope of changes

**In pynapple:** The core changes touch `base_class.py`, `time_series.py`, `interval_set.py`, `ts_group.py`, `utils.py`, and `interface_npz.py`. In each case the modification is small: adding a parameter to `__init__`, passing it through `_define_instance()`, and including it in save/load. `utils.py` gains `nap.concatenate()` and `time_origin` handling in the existing numpy concatenation and split hooks. A test file covers propagation, concatenation, backward compatibility, and error cases.


**In neurokairos:** Core IRIG decoding and clock mapping -- TTL signal processing, IRIG-H frame decoding, and the `ClockTable` dataclass. Pure numpy, no pynapple dependency. See the [neurokairos repo](https://github.com/SjulsonLab/neurokairos) for details.

**In neurokairos-pynapple:** Thin pynapple integration layer providing `BinaryFile` for lazy, memory-mapped access to binary recordings via `ClockTable`. See the [neurokairos-pynapple repo](https://github.com/SjulsonLab/neurokairos-pynapple) for details.
