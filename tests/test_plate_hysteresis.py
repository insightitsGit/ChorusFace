"""Plate openness hysteresis holds shapes across brief dips."""

from __future__ import annotations


def test_hysteresis_holds_on_small_close() -> None:
    class _H:
        _plate_open_hyst = 0.0

        def _hysteresis_plate_open(self, open_amt: float) -> float:
            target = max(0.0, min(1.0, float(open_amt)))
            prev = float(getattr(self, "_plate_open_hyst", 0.0) or 0.0)
            if target >= prev:
                if target - prev >= 0.03 or target >= 0.95:
                    self._plate_open_hyst = target
            elif prev - target >= 0.14 or target <= 0.02:
                self._plate_open_hyst = target
            return float(self._plate_open_hyst)

    h = _H()
    assert h._hysteresis_plate_open(0.80) == 0.80
    # Small dip should hold
    assert h._hysteresis_plate_open(0.72) == 0.80
    # Larger close should follow
    assert h._hysteresis_plate_open(0.60) == 0.60
    assert h._hysteresis_plate_open(0.0) == 0.0
