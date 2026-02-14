# Proposal: IRIG-H Time Synchronization for Pynapple

## The problem

Modern neuroscience experiments routinely combine data from multiple acquisition systems: electrophysiology, imaging, behavioral tracking, optogenetic stimulation, and so on. Each system records on its own clock. To analyze these streams together, researchers need to align them to a common timebase. Today, pynapple has no opinion about how timestamps relate to wall-clock time. Every object's time axis starts at some arbitrary zero, and there is no built-in way to express that two objects were recorded simultaneously or to detect when they were not.

In practice this means users align their data before loading it into pynapple, using ad-hoc scripts that vary from lab to lab. This is error-prone: silent clock drift accumulates over long recordings, off-by-one bugs in pulse counting go undetected, and there is no safety net if a user accidentally combines time series from different sessions.

## What IRIG provides

IRIG (Inter-Range Instrumentation Group) timecodes are a family of standardized timing signals widely used in instrumentation and increasingly adopted in neuroscience hardware. An IRIG encoder broadcasts a continuous pulse train that encodes UTC time. A recording system captures this pulse train alongside its neural data. By decoding the pulses, we can determine the exact UTC time corresponding to any sample in the recording, and we can measure and correct clock drift between the local oscillator and the UTC reference.

This proposal focuses on IRIG-H (one pulse per minute, 60 BCD-encoded pulses per frame), which is the format used by several common neuroscience acquisition systems. The architecture is general enough to support other IRIG variants in the future.

## What this proposal adds to pynapple

The changes fall into two layers: a small, general-purpose extension to pynapple's core data model, and a specialized processing module for IRIG decoding.

**Core layer: a `time_origin` attribute.** Every pynapple object (Tsd, TsdFrame, TsdTensor, Ts, IntervalSet, TsGroup) gains an optional `time_origin` attribute. This is a single float64 representing the Unix timestamp of the object's t=0 (the number of seconds since midnight on Jan 1, 1970). When `time_origin` is `None` (the default), the object behaves exactly as it does today -- no existing code is affected. When it is set, pynapple knows that the object's timestamps are anchored to real-world time, and it can enforce consistency: operations that combine two objects will raise an error if their time origins differ or if one is synchronized and the other is not. This catches a class of bugs that currently pass silently.

A key design choice is that timestamps remain relative. We do not convert the time axis to Unix time. Instead, `time_origin` records the offset, and the timestamps stay as small, human-readable floats (seconds from recording start). This means `restrict()`, `get()`, and manual `IntervalSet` construction continue to work with the same kinds of values users are accustomed to. A convenience method, `.origin_datetime()`, converts the origin to a Python `datetime` for display and interoperability.

`time_origin` propagates automatically. When an operation produces a new object -- `restrict()`, `count()`, `copy()`, slicing, numpy operations -- the result inherits the origin of its input. This propagation follows the existing `_define_instance()` pattern, so it requires minimal changes to the codebase and does not introduce new control flow. Save and load round-trip the attribute through NPZ files, remaining backward-compatible with files that predate the feature.

**Processing layer: IRIG decoding and drift correction.** A new module provides functions for the full synchronization workflow:

`detect_ttl_pulses()` takes a raw TTL waveform (a Tsd) and returns an IntervalSet of pulse intervals. This is a general utility -- useful beyond IRIG for any TTL-based event detection.

`decode_irig()` takes pulse intervals and decodes the IRIG-H bit stream, producing a mapping from local recording time to UTC. It handles partial frames and corrupted pulses gracefully, reporting quality metrics alongside the decoded data.

`irig_sync()` is the main user-facing function. Given a pynapple object and its associated IRIG pulse train, it decodes the timecode, fits a linear model to correct for clock drift, adjusts the object's timestamps, and sets `time_origin`. It returns the corrected object along with a diagnostic summary (drift in ppm, number of decoded frames, fit residuals).

Two instance methods complete the workflow. They correspond to two distinct steps -- **synchronization** and **alignment** -- that are worth defining clearly:

**Synchronization** means anchoring an object to an external time reference. It answers the question "when, in real-world time, did this recording start?" There are two ways to synchronize an object: `irig_sync()` (described above) decodes an IRIG pulse train and sets `time_origin` automatically, while `.set_time_origin(origin)` lets you stamp an origin manually. The latter is for objects that share a clock with an already-synced object. For example, if an electrophysiology system records both neural data and a behavior camera on the same clock, you synchronize the neural data with IRIG and then stamp the same origin onto the behavior data:

```python
ephys_synced = nap.irig_sync(ephys, irig_pulses)
behavior_synced = behavior.set_time_origin(ephys_synced.time_origin)
```

**Alignment** is the next step: shifting one synchronized object's timestamps to match another synchronized object's timebase. It answers the question "how do I put these two streams on the same clock?" Alignment requires that both objects have already been synchronized (i.e., both have a `time_origin`). Calling `A.align_to(B)` returns a new object whose timestamps have been shifted by the offset between the two origins, with `time_origin` set to match B's. The directionality is explicit -- A is being adjusted, B is the reference -- and it composes naturally when aligning multiple streams:

```python
spikes_aligned = spikes.align_to(lfp)
stim_aligned   = stim.align_to(lfp)
```

Calling `.align_to()` on an object that has no `time_origin` raises an error, because there is no offset to compute. This enforces the two-step model: synchronize first, then align.

**Concatenation.** Currently, pynapple objects can be concatenated via `np.concatenate`, which pynapple intercepts through numpy's `__array_function__` protocol. With `time_origin`, this hook gains compatibility checking (all inputs must have the same origin or all be `None`) and propagation. A new dedicated function, `nap.concatenate()`, provides the same functionality with an explicit pynapple entry point.

**Compatibility checking across all two-object operations.** Beyond the methods described above, `time_origin` compatibility is enforced wherever two pynapple objects are combined: `restrict()`, `count()`, `in_interval()`, IntervalSet set operations (`intersect()`, `union()`, `set_diff()`), and TsGroup methods (`merge()`, `value_from()`). In every case, if one object is synchronized and the other is not, or if their origins differ, the operation raises an error. When both origins are `None`, no checking occurs and behavior is identical to today.

## Why this belongs in pynapple

Time synchronization and alignment are not niche concerns. They are prerequisites for most multi-modal analyses, and getting either step wrong invalidates results. By handling both at the container level, pynapple can prevent timing errors structurally rather than relying on users to be careful. The `time_origin` attribute is cheap (one float per object, no performance impact) and invisible when unused, so it imposes no burden on users who do not need it.

The IRIG decoding module is more specialized, but IRIG timecodes are an open standard used by hardware from multiple vendors. Providing a reference decoder in pynapple saves every lab from writing their own and eliminates a source of subtle, hard-to-debug timing errors.

## Backward compatibility

Every change defaults to `None`. Existing code that does not pass `time_origin` sees no change in behavior. Existing NPZ files load without issues. The compatibility checks only activate when at least one object in an operation has a non-`None` origin. The new module introduces new public functions (`irig_sync`, `detect_ttl_pulses`, `decode_irig`, `concatenate`), and core classes gain new methods (`.set_time_origin()`, `.align_to()`, `.origin_datetime()`), but no existing method signature or behavior is modified.

## Scope of changes

The core changes touch `base_class.py`, `time_series.py`, `interval_set.py`, `ts_group.py`, `utils.py`, and `interface_npz.py`. In each case the modification is small: adding a parameter to `__init__`, passing it through `_define_instance()`, and including it in save/load. `utils.py` gains `nap.concatenate()` and `time_origin` handling in the existing numpy concatenation and split hooks. One new file, `irig.py`, contains the decoding and synchronization logic. A corresponding test file covers IRIG decoding, drift correction, propagation, concatenation, backward compatibility, and error cases.
