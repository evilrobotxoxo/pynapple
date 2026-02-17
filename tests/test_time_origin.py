"""Tests for time_origin infrastructure across all pynapple core types."""

import os
import tempfile
import warnings

import numpy as np
import pytest

import pynapple as nap


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

ORIGIN_A = 1705329000.0  # 2024-01-15 14:30:00 UTC
ORIGIN_B = 1705332600.0  # 2024-01-15 15:30:00 UTC (1 hour later)


@pytest.fixture
def tsd():
    return nap.Tsd(t=np.arange(10.0), d=np.arange(10.0))


@pytest.fixture
def tsd_synced():
    return nap.Tsd(t=np.arange(10.0), d=np.arange(10.0), time_origin=ORIGIN_A)


@pytest.fixture
def tsdframe():
    return nap.TsdFrame(
        t=np.arange(10.0), d=np.random.randn(10, 3), columns=["a", "b", "c"]
    )


@pytest.fixture
def tsdframe_synced():
    return nap.TsdFrame(
        t=np.arange(10.0),
        d=np.random.randn(10, 3),
        columns=["a", "b", "c"],
        time_origin=ORIGIN_A,
    )


@pytest.fixture
def tsdtensor():
    return nap.TsdTensor(t=np.arange(10.0), d=np.random.randn(10, 3, 4))


@pytest.fixture
def tsdtensor_synced():
    return nap.TsdTensor(
        t=np.arange(10.0), d=np.random.randn(10, 3, 4), time_origin=ORIGIN_A
    )


@pytest.fixture
def ts():
    return nap.Ts(t=np.arange(10.0))


@pytest.fixture
def ts_synced():
    return nap.Ts(t=np.arange(10.0), time_origin=ORIGIN_A)


@pytest.fixture
def ep():
    return nap.IntervalSet(start=[0, 20], end=[10, 30])


@pytest.fixture
def ep_synced():
    return nap.IntervalSet(start=[0, 20], end=[10, 30], time_origin=ORIGIN_A)


@pytest.fixture
def tsgroup():
    return nap.TsGroup(
        {0: nap.Ts(t=np.arange(10.0)), 1: nap.Ts(t=np.arange(0, 10, 0.5))},
        time_support=nap.IntervalSet(0, 10),
    )


@pytest.fixture
def tsgroup_synced():
    return nap.TsGroup(
        {0: nap.Ts(t=np.arange(10.0)), 1: nap.Ts(t=np.arange(0, 10, 0.5))},
        time_support=nap.IntervalSet(0, 10),
        time_origin=ORIGIN_A,
    )


# ===========================================================================
# 1. Construction
# ===========================================================================


class TestConstruction:
    """All types accept time_origin, default is None."""

    def test_tsd_default_none(self, tsd):
        assert tsd.time_origin is None

    def test_tsd_with_origin(self, tsd_synced):
        assert tsd_synced.time_origin == ORIGIN_A

    def test_tsdframe_default_none(self, tsdframe):
        assert tsdframe.time_origin is None

    def test_tsdframe_with_origin(self, tsdframe_synced):
        assert tsdframe_synced.time_origin == ORIGIN_A

    def test_tsdtensor_default_none(self, tsdtensor):
        assert tsdtensor.time_origin is None

    def test_tsdtensor_with_origin(self, tsdtensor_synced):
        assert tsdtensor_synced.time_origin == ORIGIN_A

    def test_ts_default_none(self, ts):
        assert ts.time_origin is None

    def test_ts_with_origin(self, ts_synced):
        assert ts_synced.time_origin == ORIGIN_A

    def test_intervalset_default_none(self, ep):
        assert ep.time_origin is None

    def test_intervalset_with_origin(self, ep_synced):
        assert ep_synced.time_origin == ORIGIN_A

    def test_tsgroup_default_none(self, tsgroup):
        assert tsgroup.time_origin is None

    def test_tsgroup_with_origin(self, tsgroup_synced):
        assert tsgroup_synced.time_origin == ORIGIN_A


# ===========================================================================
# 2. origin_datetime()
# ===========================================================================


class TestOriginDatetime:
    def test_tsd_origin_datetime(self, tsd_synced):
        import datetime

        dt = tsd_synced.origin_datetime()
        assert dt is not None
        assert dt.tzinfo == datetime.timezone.utc
        assert dt.year == 2024
        assert dt.month == 1
        assert dt.day == 15

    def test_none_returns_none(self, tsd):
        assert tsd.origin_datetime() is None

    def test_intervalset_origin_datetime(self, ep_synced):
        dt = ep_synced.origin_datetime()
        assert dt is not None

    def test_tsgroup_origin_datetime(self, tsgroup_synced):
        dt = tsgroup_synced.origin_datetime()
        assert dt is not None


# ===========================================================================
# 3. Backward compatibility — ops work with time_origin=None
# ===========================================================================


class TestBackwardCompat:
    def test_restrict(self, tsd, ep):
        result = tsd.restrict(ep)
        assert result.time_origin is None

    def test_count(self, tsd, ep):
        result = tsd.count(1.0, ep=ep)
        assert result.time_origin is None

    def test_value_from(self, tsd):
        tsd2 = nap.Tsd(t=np.arange(0, 10, 0.5), d=np.arange(20.0))
        result = tsd2.value_from(tsd)
        assert result.time_origin is None

    def test_intervalset_intersect(self, ep):
        ep2 = nap.IntervalSet(start=[5], end=[25])
        result = ep.intersect(ep2)
        assert result.time_origin is None

    def test_intervalset_union(self, ep):
        ep2 = nap.IntervalSet(start=[5], end=[25])
        result = ep.union(ep2)
        assert result.time_origin is None

    def test_intervalset_set_diff(self, ep):
        ep2 = nap.IntervalSet(start=[5], end=[25])
        result = ep.set_diff(ep2)
        assert result.time_origin is None

    def test_tsgroup_restrict(self, tsgroup, ep):
        result = tsgroup.restrict(ep)
        assert result.time_origin is None


# ===========================================================================
# 4. Compatibility checks — errors on mismatch
# ===========================================================================


class TestCompatibilityChecks:
    """TypeError for synced+unsynced, ValueError for different origins."""

    def test_restrict_synced_with_unsynced_ep(self, tsd_synced, ep):
        with pytest.raises(TypeError, match="Cannot combine"):
            tsd_synced.restrict(ep)

    def test_restrict_unsynced_with_synced_ep(self, tsd):
        ep_s = nap.IntervalSet(start=[0], end=[5], time_origin=ORIGIN_A)
        with pytest.raises(TypeError, match="Cannot combine"):
            tsd.restrict(ep_s)

    def test_restrict_different_origins(self, tsd_synced):
        ep_b = nap.IntervalSet(start=[0], end=[5], time_origin=ORIGIN_B)
        with pytest.raises(ValueError, match="different time origins"):
            tsd_synced.restrict(ep_b)

    def test_restrict_same_origin_ok(self, tsd_synced, ep_synced):
        result = tsd_synced.restrict(ep_synced)
        assert result.time_origin == ORIGIN_A

    def test_value_from_mismatch(self):
        tsd1 = nap.Tsd(t=[0, 1, 2], d=[10.0, 20.0, 30.0], time_origin=ORIGIN_A)
        tsd2 = nap.Tsd(t=[0, 1, 2], d=[1.0, 2.0, 3.0], time_origin=ORIGIN_B)
        with pytest.raises(ValueError, match="different time origins"):
            tsd1.value_from(tsd2)

    def test_value_from_synced_unsynced(self):
        tsd1 = nap.Tsd(t=[0, 1, 2], d=[10.0, 20.0, 30.0], time_origin=ORIGIN_A)
        tsd2 = nap.Tsd(t=[0, 1, 2], d=[1.0, 2.0, 3.0])
        with pytest.raises(TypeError, match="Cannot combine"):
            tsd1.value_from(tsd2)

    def test_count_ep_mismatch(self, tsd_synced):
        ep = nap.IntervalSet(start=[0], end=[5], time_origin=ORIGIN_B)
        with pytest.raises(ValueError, match="different time origins"):
            tsd_synced.count(1.0, ep=ep)

    def test_in_interval_mismatch(self, tsd_synced):
        ep = nap.IntervalSet(start=[0], end=[5])
        with pytest.raises(TypeError, match="Cannot combine"):
            tsd_synced.in_interval(ep)

    def test_intervalset_intersect_mismatch(self, ep_synced):
        ep2 = nap.IntervalSet(start=[5], end=[25])
        with pytest.raises(TypeError, match="Cannot combine"):
            ep_synced.intersect(ep2)

    def test_intervalset_union_mismatch(self, ep_synced):
        ep2 = nap.IntervalSet(start=[5], end=[25], time_origin=ORIGIN_B)
        with pytest.raises(ValueError, match="different time origins"):
            ep_synced.union(ep2)

    def test_intervalset_set_diff_mismatch(self, ep_synced):
        ep2 = nap.IntervalSet(start=[5], end=[25])
        with pytest.raises(TypeError, match="Cannot combine"):
            ep_synced.set_diff(ep2)

    def test_intervalset_in_interval_mismatch(self, ep_synced):
        ep2 = nap.IntervalSet(start=[5], end=[25])
        with pytest.raises(TypeError, match="Cannot combine"):
            ep_synced.in_interval(ep2)

    def test_tsgroup_restrict_mismatch(self, tsgroup_synced):
        ep = nap.IntervalSet(start=[0], end=[5])
        with pytest.raises(TypeError, match="Cannot combine"):
            tsgroup_synced.restrict(ep)

    def test_tsgroup_value_from_mismatch(self, tsgroup_synced):
        tsd = nap.Tsd(t=np.arange(10.0), d=np.arange(10.0))
        with pytest.raises(TypeError, match="Cannot combine"):
            tsgroup_synced.value_from(tsd)

    def test_tsgroup_count_ep_mismatch(self, tsgroup_synced):
        ep = nap.IntervalSet(start=[0], end=[5])
        with pytest.raises(TypeError, match="Cannot combine"):
            tsgroup_synced.count(1.0, ep=ep)

    def test_tsgroup_merge_group_mismatch(self, tsgroup_synced):
        other = nap.TsGroup(
            {5: nap.Ts(t=[0.0, 1.0])}, time_support=nap.IntervalSet(0, 5)
        )
        with pytest.raises(TypeError, match="Cannot combine"):
            nap.TsGroup.merge_group(tsgroup_synced, other)


# ===========================================================================
# 5. Propagation — operations preserve time_origin
# ===========================================================================


class TestPropagation:
    def test_restrict_preserves(self, tsd_synced, ep_synced):
        result = tsd_synced.restrict(ep_synced)
        assert result.time_origin == ORIGIN_A

    def test_count_preserves(self, tsd_synced):
        result = tsd_synced.count(1.0)
        assert result.time_origin == ORIGIN_A

    def test_numpy_add_preserves(self, tsd_synced):
        result = tsd_synced + 1
        assert result.time_origin == ORIGIN_A

    def test_numpy_multiply_preserves(self, tsd_synced):
        result = tsd_synced * 2
        assert result.time_origin == ORIGIN_A

    def test_numpy_slice_preserves(self, tsd_synced):
        result = tsd_synced[0:5]
        assert result.time_origin == ORIGIN_A

    def test_intervalset_intersect_preserves(self, ep_synced):
        ep2 = nap.IntervalSet(start=[5], end=[25], time_origin=ORIGIN_A)
        result = ep_synced.intersect(ep2)
        assert result.time_origin == ORIGIN_A

    def test_intervalset_union_preserves(self, ep_synced):
        ep2 = nap.IntervalSet(start=[50], end=[60], time_origin=ORIGIN_A)
        result = ep_synced.union(ep2)
        assert result.time_origin == ORIGIN_A

    def test_intervalset_set_diff_preserves(self, ep_synced):
        ep2 = nap.IntervalSet(start=[5], end=[25], time_origin=ORIGIN_A)
        result = ep_synced.set_diff(ep2)
        assert result.time_origin == ORIGIN_A

    def test_intervalset_time_span_preserves(self, ep_synced):
        result = ep_synced.time_span()
        assert result.time_origin == ORIGIN_A

    def test_intervalset_merge_close_preserves(self, ep_synced):
        result = ep_synced.merge_close_intervals(15)
        assert result.time_origin == ORIGIN_A

    def test_intervalset_split_preserves(self, ep_synced):
        result = ep_synced.split(5)
        assert result.time_origin == ORIGIN_A

    def test_intervalset_getitem_preserves(self, ep_synced):
        result = ep_synced[0]
        assert result.time_origin == ORIGIN_A

    def test_intervalset_drop_short_preserves(self, ep_synced):
        result = ep_synced.drop_short_intervals(1.0)
        assert result.time_origin == ORIGIN_A

    def test_intervalset_drop_long_preserves(self, ep_synced):
        result = ep_synced.drop_long_intervals(100.0)
        assert result.time_origin == ORIGIN_A

    def test_tsgroup_getitem_preserves(self, tsgroup_synced):
        result = tsgroup_synced[[0]]
        assert result.time_origin == ORIGIN_A

    def test_tsgroup_restrict_preserves(self, tsgroup_synced):
        ep = nap.IntervalSet(start=[0], end=[5], time_origin=ORIGIN_A)
        result = tsgroup_synced.restrict(ep)
        assert result.time_origin == ORIGIN_A

    def test_tsgroup_get_preserves(self, tsgroup_synced):
        result = tsgroup_synced.get(0, 5)
        assert result.time_origin == ORIGIN_A

    def test_tsgroup_merge_preserves(self, tsgroup_synced):
        other = nap.TsGroup(
            {5: nap.Ts(t=[0.0, 1.0])},
            time_support=nap.IntervalSet(0, 5),
            time_origin=ORIGIN_A,
        )
        result = tsgroup_synced.merge_group(other)
        assert result.time_origin == ORIGIN_A

    def test_concatenate_tsd_hook(self, tsd_synced):
        """np.concatenate on Tsd arrays preserves time_origin."""
        tsd2 = nap.Tsd(
            t=np.arange(10.0, 20.0), d=np.arange(10.0, 20.0), time_origin=ORIGIN_A
        )
        result = np.concatenate((tsd_synced, tsd2))
        assert result.time_origin == ORIGIN_A

    def test_ts_define_instance_preserves(self, ts_synced):
        """Ts slicing preserves time_origin."""
        result = ts_synced[0:5]
        assert result.time_origin == ORIGIN_A


# ===========================================================================
# 6. set_time_origin()
# ===========================================================================


class TestSetTimeOrigin:
    def test_tsd_set_time_origin(self, tsd):
        result = tsd.set_time_origin(ORIGIN_A)
        assert result.time_origin == ORIGIN_A
        assert tsd.time_origin is None  # original unchanged
        np.testing.assert_array_equal(result.times(), tsd.times())
        np.testing.assert_array_equal(result.values, tsd.values)

    def test_ts_set_time_origin(self, ts):
        result = ts.set_time_origin(ORIGIN_A)
        assert result.time_origin == ORIGIN_A
        assert ts.time_origin is None
        np.testing.assert_array_equal(result.times(), ts.times())

    def test_tsdframe_set_time_origin(self, tsdframe):
        result = tsdframe.set_time_origin(ORIGIN_A)
        assert result.time_origin == ORIGIN_A
        assert tsdframe.time_origin is None

    def test_tsdtensor_set_time_origin(self, tsdtensor):
        result = tsdtensor.set_time_origin(ORIGIN_A)
        assert result.time_origin == ORIGIN_A
        assert tsdtensor.time_origin is None

    def test_intervalset_set_time_origin(self, ep):
        result = ep.set_time_origin(ORIGIN_A)
        assert result.time_origin == ORIGIN_A
        assert ep.time_origin is None
        np.testing.assert_array_equal(result.start, ep.start)
        np.testing.assert_array_equal(result.end, ep.end)

    def test_tsgroup_set_time_origin(self, tsgroup):
        result = tsgroup.set_time_origin(ORIGIN_A)
        assert result.time_origin == ORIGIN_A
        assert tsgroup.time_origin is None

    def test_clear_time_origin(self, tsd_synced):
        result = tsd_synced.set_time_origin(None)
        assert result.time_origin is None

    def test_invalid_type_raises(self, tsd):
        with pytest.raises(TypeError):
            tsd.set_time_origin("not a number")


# ===========================================================================
# 7. sync_to()
# ===========================================================================


class TestSyncTo:
    def test_tsd_sync_to(self):
        """Synchronizing shifts timestamps by offset = self.origin - other.origin."""
        tsd1 = nap.Tsd(t=[0.0, 1.0, 2.0], d=[10.0, 20.0, 30.0], time_origin=ORIGIN_A)
        tsd2 = nap.Tsd(t=[0.0, 1.0], d=[1.0, 2.0], time_origin=ORIGIN_B)
        offset = ORIGIN_A - ORIGIN_B  # -3600

        result = tsd1.sync_to(tsd2)
        np.testing.assert_array_almost_equal(
            result.times(), np.array([0.0, 1.0, 2.0]) + offset
        )
        assert result.time_origin == ORIGIN_B

    def test_ts_sync_to(self):
        ts1 = nap.Ts(t=[0.0, 1.0], time_origin=ORIGIN_A)
        ts2 = nap.Ts(t=[0.0], time_origin=ORIGIN_B)
        result = ts1.sync_to(ts2)
        assert result.time_origin == ORIGIN_B

    def test_intervalset_sync_to(self):
        ep1 = nap.IntervalSet(start=[0, 20], end=[10, 30], time_origin=ORIGIN_A)
        ep2 = nap.IntervalSet(start=[0], end=[5], time_origin=ORIGIN_B)
        offset = ORIGIN_A - ORIGIN_B

        result = ep1.sync_to(ep2)
        np.testing.assert_array_almost_equal(
            result.start, np.array([0, 20]) + offset
        )
        np.testing.assert_array_almost_equal(result.end, np.array([10, 30]) + offset)
        assert result.time_origin == ORIGIN_B

    def test_tsgroup_sync_to(self):
        tsg = nap.TsGroup(
            {0: nap.Ts(t=[0.0, 1.0, 2.0])},
            time_support=nap.IntervalSet(0, 5),
            time_origin=ORIGIN_A,
        )
        other = nap.Ts(t=[0.0], time_origin=ORIGIN_B)
        offset = ORIGIN_A - ORIGIN_B

        result = tsg.sync_to(other)
        assert result.time_origin == ORIGIN_B
        np.testing.assert_array_almost_equal(
            result[0].times(), np.array([0.0, 1.0, 2.0]) + offset
        )

    def test_sync_to_no_origin_self_raises(self, tsd):
        other = nap.Tsd(t=[0.0], d=[1.0], time_origin=ORIGIN_A)
        with pytest.raises(TypeError, match="self has no time_origin"):
            tsd.sync_to(other)

    def test_sync_to_no_origin_other_raises(self, tsd_synced, tsd):
        with pytest.raises(TypeError, match="other has no time_origin"):
            tsd_synced.sync_to(tsd)


# ===========================================================================
# 8. nap.concatenate()
# ===========================================================================


class TestConcatenate:
    # --- Time series ---

    def test_concat_tsd(self):
        tsd1 = nap.Tsd(t=[0.0, 1.0], d=[10.0, 20.0])
        tsd2 = nap.Tsd(t=[3.0, 4.0], d=[30.0, 40.0])
        result = nap.concatenate(tsd1, tsd2)
        assert isinstance(result, nap.Tsd)
        np.testing.assert_array_equal(result.times(), [0, 1, 3, 4])
        np.testing.assert_array_equal(result.values, [10, 20, 30, 40])

    def test_concat_tsd_with_origin(self):
        tsd1 = nap.Tsd(t=[0.0, 1.0], d=[10.0, 20.0], time_origin=ORIGIN_A)
        tsd2 = nap.Tsd(t=[3.0, 4.0], d=[30.0, 40.0], time_origin=ORIGIN_A)
        result = nap.concatenate(tsd1, tsd2)
        assert result.time_origin == ORIGIN_A

    def test_concat_tsd_unsorted_warns(self):
        tsd1 = nap.Tsd(t=[5.0, 6.0], d=[50.0, 60.0])
        tsd2 = nap.Tsd(t=[0.0, 1.0], d=[10.0, 20.0])
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = nap.concatenate(tsd1, tsd2)
            sort_warnings = [x for x in w if "not sorted" in str(x.message)]
            assert len(sort_warnings) > 0
        np.testing.assert_array_equal(result.times(), [0, 1, 5, 6])

    def test_concat_tsd_duplicate_raises(self):
        tsd1 = nap.Tsd(t=[0.0, 1.0], d=[10.0, 20.0])
        tsd2 = nap.Tsd(t=[1.0, 2.0], d=[30.0, 40.0])
        with pytest.raises(ValueError, match="Duplicate timestamps"):
            nap.concatenate(tsd1, tsd2)

    def test_concat_ts(self):
        ts1 = nap.Ts(t=[0.0, 1.0])
        ts2 = nap.Ts(t=[5.0, 6.0])
        result = nap.concatenate(ts1, ts2)
        assert isinstance(result, nap.Ts)
        np.testing.assert_array_equal(result.times(), [0, 1, 5, 6])

    def test_concat_tsdframe(self):
        d1 = np.array([[1.0, 2.0], [3.0, 4.0]])
        d2 = np.array([[5.0, 6.0], [7.0, 8.0]])
        tdf1 = nap.TsdFrame(t=[0.0, 1.0], d=d1, columns=["a", "b"])
        tdf2 = nap.TsdFrame(t=[3.0, 4.0], d=d2, columns=["a", "b"])
        result = nap.concatenate(tdf1, tdf2)
        assert isinstance(result, nap.TsdFrame)
        assert list(result.columns) == ["a", "b"]

    def test_concat_tsdtensor(self):
        d1 = np.zeros((2, 3, 4))
        d2 = np.ones((3, 3, 4))
        tst1 = nap.TsdTensor(t=[0.0, 1.0], d=d1)
        tst2 = nap.TsdTensor(t=[3.0, 4.0, 5.0], d=d2)
        result = nap.concatenate(tst1, tst2)
        assert isinstance(result, nap.TsdTensor)
        assert result.shape == (5, 3, 4)

    def test_concat_as_list(self):
        tsd1 = nap.Tsd(t=[0.0, 1.0], d=[10.0, 20.0])
        tsd2 = nap.Tsd(t=[3.0, 4.0], d=[30.0, 40.0])
        result = nap.concatenate([tsd1, tsd2])
        np.testing.assert_array_equal(result.times(), [0, 1, 3, 4])

    # --- IntervalSet ---

    def test_concat_intervalset(self):
        ep1 = nap.IntervalSet(start=[0], end=[5])
        ep2 = nap.IntervalSet(start=[10], end=[15])
        result = nap.concatenate(ep1, ep2)
        assert isinstance(result, nap.IntervalSet)
        assert len(result) == 2

    def test_concat_intervalset_with_origin(self):
        ep1 = nap.IntervalSet(start=[0], end=[5], time_origin=ORIGIN_A)
        ep2 = nap.IntervalSet(start=[10], end=[15], time_origin=ORIGIN_A)
        result = nap.concatenate(ep1, ep2)
        assert result.time_origin == ORIGIN_A

    # --- TsGroup ---

    def test_concat_tsgroup(self):
        tsg1 = nap.TsGroup(
            {0: nap.Ts(t=[0.0, 1.0]), 1: nap.Ts(t=[0.5])},
            time_support=nap.IntervalSet(0, 3),
        )
        tsg2 = nap.TsGroup(
            {0: nap.Ts(t=[5.0, 6.0]), 2: nap.Ts(t=[5.5])},
            time_support=nap.IntervalSet(5, 7),
        )
        result = nap.concatenate(tsg1, tsg2)
        assert isinstance(result, nap.TsGroup)
        assert set(result.keys()) == {0, 1, 2}
        # Key 0 should have merged timestamps
        np.testing.assert_array_equal(result[0].times(), [0, 1, 5, 6])

    def test_concat_tsgroup_with_origin(self):
        tsg1 = nap.TsGroup(
            {0: nap.Ts(t=[0.0, 1.0])},
            time_support=nap.IntervalSet(0, 3),
            time_origin=ORIGIN_A,
        )
        tsg2 = nap.TsGroup(
            {0: nap.Ts(t=[5.0, 6.0])},
            time_support=nap.IntervalSet(5, 7),
            time_origin=ORIGIN_A,
        )
        result = nap.concatenate(tsg1, tsg2)
        assert result.time_origin == ORIGIN_A

    # --- Error cases ---

    def test_concat_empty_raises(self):
        with pytest.raises(ValueError, match="No objects"):
            nap.concatenate()

    def test_concat_type_mismatch_raises(self):
        tsd = nap.Tsd(t=[0.0], d=[1.0])
        ep = nap.IntervalSet(start=[0], end=[1])
        with pytest.raises(TypeError, match="same type"):
            nap.concatenate(tsd, ep)

    def test_concat_origin_mismatch_raises(self):
        tsd1 = nap.Tsd(t=[0.0], d=[1.0], time_origin=ORIGIN_A)
        tsd2 = nap.Tsd(t=[1.0], d=[2.0], time_origin=ORIGIN_B)
        with pytest.raises(ValueError, match="different time origins"):
            nap.concatenate(tsd1, tsd2)

    def test_concat_synced_unsynced_raises(self):
        tsd1 = nap.Tsd(t=[0.0], d=[1.0], time_origin=ORIGIN_A)
        tsd2 = nap.Tsd(t=[1.0], d=[2.0])
        with pytest.raises(TypeError, match="UTC time-referenced and unreferenced"):
            nap.concatenate(tsd1, tsd2)

    def test_concat_single_returns_same(self):
        tsd = nap.Tsd(t=[0.0], d=[1.0])
        result = nap.concatenate(tsd)
        assert result is tsd


# ===========================================================================
# 9. Save/load round-trip
# ===========================================================================


class TestSaveLoad:
    def _save_load(self, obj, filename):
        tmpdir = tempfile.mkdtemp()
        path = os.path.join(tmpdir, filename)
        obj.save(path)
        loaded = nap.load_file(path)
        return loaded

    def test_tsd_save_load(self, tsd_synced):
        loaded = self._save_load(tsd_synced, "tsd.npz")
        assert loaded.time_origin == ORIGIN_A
        np.testing.assert_array_equal(loaded.times(), tsd_synced.times())
        np.testing.assert_array_equal(loaded.values, tsd_synced.values)

    def test_tsd_save_load_no_origin(self, tsd):
        loaded = self._save_load(tsd, "tsd.npz")
        assert loaded.time_origin is None

    def test_tsdframe_save_load(self, tsdframe_synced):
        loaded = self._save_load(tsdframe_synced, "tsdframe.npz")
        assert loaded.time_origin == ORIGIN_A

    def test_tsdtensor_save_load(self, tsdtensor_synced):
        loaded = self._save_load(tsdtensor_synced, "tsdtensor.npz")
        assert loaded.time_origin == ORIGIN_A

    def test_ts_save_load(self, ts_synced):
        loaded = self._save_load(ts_synced, "ts.npz")
        assert loaded.time_origin == ORIGIN_A

    def test_intervalset_save_load(self, ep_synced):
        loaded = self._save_load(ep_synced, "ep.npz")
        assert loaded.time_origin == ORIGIN_A
        np.testing.assert_array_equal(loaded.start, ep_synced.start)
        np.testing.assert_array_equal(loaded.end, ep_synced.end)

    def test_tsgroup_save_load(self, tsgroup_synced):
        loaded = self._save_load(tsgroup_synced, "tsgroup.npz")
        assert loaded.time_origin == ORIGIN_A
        assert set(loaded.keys()) == set(tsgroup_synced.keys())


# ===========================================================================
# 10. __repr__ sync indicator
# ===========================================================================


class TestRepr:
    def test_tsd_repr_shows_origin(self, tsd_synced):
        r = repr(tsd_synced)
        assert "time_origin:" in r
        assert "UTC" in r

    def test_tsd_repr_no_origin(self, tsd):
        r = repr(tsd)
        assert "time_origin" not in r

    def test_tsdframe_repr_shows_origin(self, tsdframe_synced):
        r = repr(tsdframe_synced)
        assert "time_origin:" in r

    def test_tsdtensor_repr_shows_origin(self, tsdtensor_synced):
        r = repr(tsdtensor_synced)
        assert "time_origin:" in r

    def test_ts_repr_shows_origin(self, ts_synced):
        r = repr(ts_synced)
        assert "time_origin:" in r

    def test_intervalset_repr_shows_origin(self, ep_synced):
        r = repr(ep_synced)
        assert "time_origin:" in r

    def test_tsgroup_repr_shows_origin(self, tsgroup_synced):
        r = repr(tsgroup_synced)
        assert "time_origin:" in r

    def test_intervalset_repr_no_origin(self, ep):
        r = repr(ep)
        assert "time_origin" not in r

    def test_tsgroup_repr_no_origin(self, tsgroup):
        r = repr(tsgroup)
        assert "time_origin" not in r
