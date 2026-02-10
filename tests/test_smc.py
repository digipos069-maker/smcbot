import numpy as np
import pandas as pd

from main import detect_swings, detect_fvg, detect_structure_breaks, generate_signals, SMCConfig


def _df_from_cols(open_, high, low, close):
    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
        }
    )


def test_detect_swings_simple():
    df = _df_from_cols(
        open_=[10, 12, 11, 12, 11],
        high=[11, 14, 12, 13, 12],
        low=[9, 10, 8, 10, 9],
        close=[10, 13, 9, 12, 11],
    )
    swing_high, swing_low = detect_swings(df, n=1)
    assert bool(swing_high[1]) is True
    assert bool(swing_low[2]) is True


def test_detect_fvg_bullish_and_bearish():
    df = _df_from_cols(
        open_=[10, 10, 10, 10],
        high=[11, 12, 9, 8],
        low=[9, 10, 8, 7],
        close=[10, 11, 9, 7.5],
    )
    bullish, bearish = detect_fvg(df, min_gap=0.0)
    # bullish at i=2: low[2] (8) above high[0] (11) -> false
    # bearish at i=3: high[3] (8) below low[1] (10) -> true
    assert bullish[2] is None
    assert bearish[3] == (df["high"].iloc[3], df["low"].iloc[1])


def test_structure_breaks_no_double_trigger():
    df = _df_from_cols(
        open_=[10, 10, 10],
        high=[10, 12, 12],
        low=[9, 11, 11],
        close=[10, 11, 10.5],
    )
    swing_high = np.array([True, False, False])
    swing_low = np.array([False, True, False])
    # last_high=10, last_low=11, close[2]=11 triggers cl>last_high and cl<last_low simultaneously in old logic
    bos_bull, bos_bear, choch_bull, choch_bear = detect_structure_breaks(df, swing_high, swing_low)
    assert not (bos_bull[2] and bos_bear[2])
    assert not (choch_bull[2] and choch_bear[2])


def test_generate_signals_requires_tap_after_fvg():
    df = _df_from_cols(
        open_=[11, 12, 12, 11, 10.5, 9, 14.8, 17.2, 16.7],
        high=[12, 15, 13, 12, 12, 16.2, 15, 18, 17.2],
        low=[11, 11, 10, 9.2, 9.0, 8.5, 14, 17, 16.5],
        close=[11.5, 12, 11, 9.5, 11, 16.3, 14.5, 17.5, 16.8],
    )
    cfg = SMCConfig(swing_n=1, choch_lookahead=10, fvg_lookahead=10, min_gap=0.0)
    out = generate_signals(df, cfg)
    sig = out["signals"]
    assert sig[7] != "BUY"
    assert sig[8] == "BUY"
