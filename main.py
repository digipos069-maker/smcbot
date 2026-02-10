import sys
from dataclasses import dataclass
from typing import Optional, List, Tuple

import numpy as np
import pandas as pd

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QFileDialog, QLabel, QSpinBox, QDoubleSpinBox, QMessageBox,
    QComboBox, QDateEdit, QCheckBox, QTabWidget, QDialog, QTableWidget,
    QTableWidgetItem, QHeaderView
)
from PyQt5.QtCore import Qt, QDate

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import os
from datetime import datetime, time

try:
    import MetaTrader5 as mt5
except Exception:
    mt5 = None


# ----------------------------
# SMC RULES (STRICT + TESTABLE)
# ----------------------------

@dataclass
class SMCConfig:
    swing_n: int = 2          # fractal window
    choch_lookahead: int = 12 # candles after sweep to allow CHOCH
    fvg_lookahead: int = 20   # candles after CHOCH to allow FVG touch
    min_gap: float = 0.0      # min FVG size (in price units)


def detect_swings(df: pd.DataFrame, n: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return boolean arrays: swing_high, swing_low."""
    highs = df["high"].to_numpy()
    lows = df["low"].to_numpy()

    swing_high = np.zeros(len(df), dtype=bool)
    swing_low = np.zeros(len(df), dtype=bool)

    for i in range(n, len(df) - n):
        window_h = highs[i - n:i + n + 1]
        window_l = lows[i - n:i + n + 1]
        if highs[i] == window_h.max() and (window_h.argmax() == n):
            swing_high[i] = True
        if lows[i] == window_l.min() and (window_l.argmin() == n):
            swing_low[i] = True

    return swing_high, swing_low


def detect_fvg(df: pd.DataFrame, min_gap: float = 0.0):
    """Return arrays of FVG zones:
       bullish_fvg[i] = (low[i], high[i-2]) if bullish gap exists else None
       bearish_fvg[i] = (low[i-2], high[i]) if bearish gap exists else None
    """
    hi = df["high"].to_numpy()
    lo = df["low"].to_numpy()

    bullish = [None] * len(df)
    bearish = [None] * len(df)

    for i in range(2, len(df)):
        # Bullish FVG: candle i low above candle i-2 high
        gap = lo[i] - hi[i - 2]
        if gap > min_gap:
            bullish[i] = (hi[i - 2], lo[i])  # zone: [high(i-2), low(i)]
        # Bearish FVG: candle i high below candle i-2 low
        gap2 = lo[i - 2] - hi[i]
        if gap2 > min_gap:
            bearish[i] = (hi[i], lo[i - 2])  # zone: [high(i), low(i-2)]

    return bullish, bearish


def detect_sweeps(df: pd.DataFrame, swing_high: np.ndarray, swing_low: np.ndarray):
    """Buy-side sweep: wick above a prior swing high, but close below it.
       Sell-side sweep: wick below a prior swing low, but close above it.
    """
    hi = df["high"].to_numpy()
    lo = df["low"].to_numpy()
    cl = df["close"].to_numpy()

    last_swing_high_price = None
    last_swing_low_price = None

    buy_sweep = np.zeros(len(df), dtype=bool)
    sell_sweep = np.zeros(len(df), dtype=bool)

    for i in range(len(df)):
        if swing_high[i]:
            last_swing_high_price = df["high"].iloc[i]
        if swing_low[i]:
            last_swing_low_price = df["low"].iloc[i]

        # Sweep rules
        if last_swing_high_price is not None:
            if hi[i] > last_swing_high_price and cl[i] < last_swing_high_price:
                buy_sweep[i] = True

        if last_swing_low_price is not None:
            if lo[i] < last_swing_low_price and cl[i] > last_swing_low_price:
                sell_sweep[i] = True

    return buy_sweep, sell_sweep


def detect_structure_breaks(df: pd.DataFrame, swing_high: np.ndarray, swing_low: np.ndarray):
    """Compute BOS/CHOCH events based on breaking last swing points."""
    cl = df["close"].to_numpy()

    last_high = None
    last_low = None
    bias = None  # 'bull' or 'bear'

    bos_bull = np.zeros(len(df), dtype=bool)
    bos_bear = np.zeros(len(df), dtype=bool)
    choch_bear = np.zeros(len(df), dtype=bool)  # bearish CHOCH (break down while bull)
    choch_bull = np.zeros(len(df), dtype=bool)  # bullish CHOCH (break up while bear)

    for i in range(len(df)):
        if swing_high[i]:
            last_high = df["high"].iloc[i]
        if swing_low[i]:
            last_low = df["low"].iloc[i]

        if last_high is not None and cl[i] > last_high:
            # break up
            if bias == "bear":
                choch_bull[i] = True
                bias = "bull"
            else:
                bos_bull[i] = True
                bias = "bull"
        elif last_low is not None and cl[i] < last_low:
            # break down
            if bias == "bull":
                choch_bear[i] = True
                bias = "bear"
            else:
                bos_bear[i] = True
                bias = "bear"

    return bos_bull, bos_bear, choch_bull, choch_bear


def generate_signals(df: pd.DataFrame, cfg: SMCConfig):
    """Rule-based signals:
    - SELL: buy-side sweep -> bearish CHOCH within choch_lookahead -> price taps bearish FVG zone within fvg_lookahead
    - BUY: sell-side sweep -> bullish CHOCH -> taps bullish FVG
    """
    swing_high, swing_low = detect_swings(df, cfg.swing_n)
    bullish_fvg, bearish_fvg = detect_fvg(df, cfg.min_gap)
    buy_sweep, sell_sweep = detect_sweeps(df, swing_high, swing_low)
    bos_bull, bos_bear, choch_bull, choch_bear = detect_structure_breaks(df, swing_high, swing_low)

    signals = np.array([""] * len(df), dtype=object)

    # Helper: candle touches a zone [a,b] if its range overlaps it
    hi = df["high"].to_numpy()
    lo = df["low"].to_numpy()

    # Track events
    buy_sweep_idx = np.where(buy_sweep)[0]
    sell_sweep_idx = np.where(sell_sweep)[0]
    choch_bear_idx = np.where(choch_bear)[0]
    choch_bull_idx = np.where(choch_bull)[0]

    # SELL pipeline: buy_sweep -> choch_bear -> tap bearish_fvg
    for s in buy_sweep_idx:
        # find first bearish CHOCH after sweep within lookahead
        candidates = choch_bear_idx[(choch_bear_idx > s) & (choch_bear_idx <= s + cfg.choch_lookahead)]
        if len(candidates) == 0:
            continue
        c = candidates[0]

        # find a bearish FVG after CHOCH and wait for a later tap
        active_zones = []
        for i in range(c + 1, min(len(df), c + 1 + cfg.fvg_lookahead)):
            zone = bearish_fvg[i]
            if zone is None:
                pass
            else:
                z_low, z_high = zone  # zone boundaries
                active_zones.append((i, z_low, z_high))

            # tap condition: candle range overlaps any prior zone
            tapped = False
            for zi, z_low, z_high in active_zones:
                if i <= zi:
                    continue
                if not (hi[i] < z_low or lo[i] > z_high):
                    signals[i] = "SELL"
                    tapped = True
                    break
            if tapped:
                break

    # BUY pipeline: sell_sweep -> choch_bull -> tap bullish_fvg
    for s in sell_sweep_idx:
        candidates = choch_bull_idx[(choch_bull_idx > s) & (choch_bull_idx <= s + cfg.choch_lookahead)]
        if len(candidates) == 0:
            continue
        c = candidates[0]

        active_zones = []
        for i in range(c + 1, min(len(df), c + 1 + cfg.fvg_lookahead)):
            zone = bullish_fvg[i]
            if zone is None:
                pass
            else:
                z_low, z_high = zone
                active_zones.append((i, z_low, z_high))

            tapped = False
            for zi, z_low, z_high in active_zones:
                if i <= zi:
                    continue
                if not (hi[i] < z_low or lo[i] > z_high):
                    signals[i] = "BUY"
                    tapped = True
                    break
            if tapped:
                break

    return {
        "swing_high": swing_high,
        "swing_low": swing_low,
        "buy_sweep": buy_sweep,
        "sell_sweep": sell_sweep,
        "choch_bull": choch_bull,
        "choch_bear": choch_bear,
        "signals": signals,
    }


def backtest_simple(
    df: pd.DataFrame,
    signals: np.ndarray,
    lot: float,
    contract_size: float,
    init_balance: float,
    leverage: float,
):
    """Simple backtest:
    - Enter on signal candle close
    - Exit on next opposite signal close (or end)
    - PnL = price_diff * lot * contract_size (quote currency)
    - Stop trading when balance <= 0
    - Enforce margin: required = (contract_size * lot * entry_price) / leverage
    """
    closes = df["close"].to_numpy()
    trades = []
    pos = None  # ("BUY"|"SELL", entry_idx, entry_price)
    balance = float(init_balance)
    skipped_entries = 0

    for i, sig in enumerate(signals):
        if balance <= 0:
            break
        if sig not in ("BUY", "SELL"):
            continue

        if pos is None:
            required_margin = (contract_size * lot * closes[i]) / leverage if leverage > 0 else float("inf")
            if balance < required_margin:
                skipped_entries += 1
                continue
            pos = (sig, i, closes[i])
            continue

        side, entry_i, entry_p = pos
        if sig == side:
            continue

        exit_p = closes[i]
        pnl = (exit_p - entry_p) * lot * contract_size if side == "BUY" else (entry_p - exit_p) * lot * contract_size
        balance += pnl
        trades.append((side, entry_i, i, entry_p, exit_p, pnl, balance))
        required_margin = (contract_size * lot * closes[i]) / leverage if leverage > 0 else float("inf")
        if balance < required_margin:
            skipped_entries += 1
            pos = None
        else:
            pos = (sig, i, closes[i])

    if pos is not None:
        side, entry_i, entry_p = pos
        exit_p = closes[-1]
        pnl = (exit_p - entry_p) * lot * contract_size if side == "BUY" else (entry_p - exit_p) * lot * contract_size
        balance += pnl
        trades.append((side, entry_i, len(closes) - 1, entry_p, exit_p, pnl, balance))

    total = float(np.sum([t[5] for t in trades])) if trades else 0.0
    wins = sum(1 for t in trades if t[5] > 0)
    losses = sum(1 for t in trades if t[5] < 0)
    gross_profit = float(np.sum([t[5] for t in trades if t[5] > 0])) if trades else 0.0
    gross_loss = float(np.sum([-t[5] for t in trades if t[5] < 0])) if trades else 0.0
    return trades, total, wins, losses, gross_profit, gross_loss, balance, skipped_entries


# ----------------------------
# Plotting (candles + signals)
# ----------------------------

def plot_candles(ax, df: pd.DataFrame, max_bars: int = 400):
    """Simple candle plot with matplotlib primitives (no extra libs)."""
    d = df.copy()
    if len(d) > max_bars:
        d = d.iloc[-max_bars:].copy()

    o = d["open"].to_numpy()
    h = d["high"].to_numpy()
    l = d["low"].to_numpy()
    c = d["close"].to_numpy()
    x = np.arange(len(d))

    for i in range(len(d)):
        # wick
        ax.vlines(x[i], l[i], h[i], linewidth=1)
        # body
        bottom = min(o[i], c[i])
        height = abs(c[i] - o[i])
        if height == 0:
            height = (h[i] - l[i]) * 0.02  # tiny body
        ax.add_patch(
            __import__("matplotlib").patches.Rectangle(
                (x[i] - 0.32, bottom),
                0.64,
                height,
                fill=False,
                linewidth=1
            )
        )

    ax.set_xlim(-1, len(d))
    ax.set_title("SMC Rule-Based Signals (Candles)")
    ax.grid(True, alpha=0.25)
    return d


def plot_markers(ax, d: pd.DataFrame, signals: np.ndarray):
    """Plot BUY/SELL markers on recent window data."""
    max_bars = len(d)
    sig = signals[-max_bars:]
    x = np.arange(max_bars)

    buys = np.where(sig == "BUY")[0]
    sells = np.where(sig == "SELL")[0]

    if len(buys) > 0:
        ax.scatter(x[buys], d["low"].to_numpy()[buys], marker="^", s=80)
    if len(sells) > 0:
        ax.scatter(x[sells], d["high"].to_numpy()[sells], marker="v", s=80)


# ----------------------------
# UI (PyQt5)
# ----------------------------

class MplCanvas(FigureCanvas):
    def __init__(self):
        self.fig = Figure(figsize=(10, 5), dpi=100)
        self.ax = self.fig.add_subplot(111)
        super().__init__(self.fig)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SMC Rule-Based System (PyQt5)")
        self.df: Optional[pd.DataFrame] = None
        self.smc_out = None

        # Controls
        self.btn_load = QPushButton("Load History Data")
        self.btn_run = QPushButton("Run SMC Rules")
        self.btn_run.setEnabled(False)

        self.symbol = QComboBox()
        self.symbol.setEditable(True)
        self.symbol.addItems(["XAUUSDm", "EURUSDm", "XAUUSD", "EURUSD"])

        self.timeframe = QComboBox()
        self.timeframe.addItems(["M5", "M15", "H1", "H2", "H4", "D1"])

        self.date_from = QDateEdit()
        self.date_from.setCalendarPopup(True)
        self.date_from.setDate(QDate.currentDate().addMonths(-1))

        self.date_to = QDateEdit()
        self.date_to.setCalendarPopup(True)
        self.date_to.setDate(QDate.currentDate())

        self.swing_n = QSpinBox()
        self.swing_n.setRange(1, 10)
        self.swing_n.setValue(2)

        self.choch_k = QSpinBox()
        self.choch_k.setRange(1, 200)
        self.choch_k.setValue(12)

        self.fvg_k = QSpinBox()
        self.fvg_k.setRange(1, 400)
        self.fvg_k.setValue(20)

        self.min_gap = QDoubleSpinBox()
        self.min_gap.setDecimals(6)
        self.min_gap.setRange(0.0, 999999.0)
        self.min_gap.setValue(0.0)

        self.lot = QDoubleSpinBox()
        self.lot.setDecimals(2)
        self.lot.setRange(0.01, 100.0)
        self.lot.setValue(0.10)

        self.contract_size = QDoubleSpinBox()
        self.contract_size.setDecimals(2)
        self.contract_size.setRange(1.0, 10000000.0)
        self.contract_size.setValue(100.0)

        self.leverage = QComboBox()
        self.leverage.addItems(["10", "50", "100", "200", "500", "1000", "2000"])
        self.leverage.setCurrentText("100")

        self.init_balance = QDoubleSpinBox()
        self.init_balance.setDecimals(2)
        self.init_balance.setRange(0.0, 100000000.0)
        self.init_balance.setValue(1000.0)

        self.quote_usd = QCheckBox("Quote in USD")
        self.quote_usd.setChecked(True)

        self.status = QLabel("Load OHLC CSV to begin.")
        self.status.setWordWrap(True)
        self.bt_status = QLabel("")
        self.bt_status.setWordWrap(True)
        self.btn_report = QPushButton("Show Report")

        # Chart
        self.canvas = MplCanvas()

        # Layout
        top = QWidget()
        self.setCentralWidget(top)
        root = QVBoxLayout(top)
        tabs = QTabWidget()
        root.addWidget(tabs)

        # Tabs
        tab_trading = QWidget()
        tab_backtest = QWidget()
        tab_settings = QWidget()
        tabs.addTab(tab_trading, "Trading")
        tabs.addTab(tab_backtest, "Backtest")
        tabs.addTab(tab_settings, "Settings")

        # Backtest layout (current page)
        backtest_root = QVBoxLayout(tab_backtest)

        row = QHBoxLayout()
        row.addWidget(self.btn_load)
        row.addWidget(self.btn_run)

        row.addWidget(QLabel("Symbol"))
        row.addWidget(self.symbol)
        row.addWidget(QLabel("Timeframe"))
        row.addWidget(self.timeframe)
        row.addWidget(QLabel("From"))
        row.addWidget(self.date_from)
        row.addWidget(QLabel("To"))
        row.addWidget(self.date_to)

        row.addWidget(QLabel("Swing n"))
        row.addWidget(self.swing_n)
        row.addWidget(QLabel("CHOCH lookahead"))
        row.addWidget(self.choch_k)
        row.addWidget(QLabel("FVG lookahead"))
        row.addWidget(self.fvg_k)
        row.addWidget(QLabel("Min gap"))
        row.addWidget(self.min_gap)
        row.addWidget(QLabel("Lot"))
        row.addWidget(self.lot)
        row.addWidget(QLabel("Contract size"))
        row.addWidget(self.contract_size)
        row.addWidget(QLabel("Leverage"))
        row.addWidget(self.leverage)
        row.addWidget(QLabel("Balance"))
        row.addWidget(self.init_balance)
        row.addWidget(self.quote_usd)
        row.addStretch(1)

        backtest_root.addLayout(row)
        backtest_root.addWidget(self.status)
        backtest_root.addWidget(self.canvas)
        bt_row = QHBoxLayout()
        bt_row.addWidget(self.bt_status, stretch=1)
        bt_row.addWidget(self.btn_report)
        backtest_root.addLayout(bt_row)

        # Trading tab placeholder
        trading_root = QVBoxLayout(tab_trading)
        trading_root.addWidget(QLabel("Trading tab (coming soon)."))

        # Settings tab placeholder
        settings_root = QVBoxLayout(tab_settings)
        settings_root.addWidget(QLabel("Settings tab (coming soon)."))

        # Signals
        self.btn_load.clicked.connect(self.load_mt5_history)
        self.btn_run.clicked.connect(self.run_rules)
        self.btn_report.clicked.connect(self.show_report)

    def show_report(self):
        if self.df is None or self.smc_out is None:
            QMessageBox.information(self, "Report", "Run a backtest first.")
            return

        sig = self.smc_out["signals"]
        init_bal = float(self.init_balance.value())
        leverage = float(self.leverage.currentText())
        trades, total, wins, losses, gross_profit, gross_loss, end_balance, skipped_entries = backtest_simple(
            self.df,
            sig,
            float(self.lot.value()),
            float(self.contract_size.value()),
            init_bal,
            leverage,
        )
        dlg = QDialog(self)
        dlg.setWindowTitle("Backtest Trade Report")
        dlg.resize(900, 500)

        layout = QVBoxLayout(dlg)
        table = QTableWidget()
        table.setColumnCount(7)
        table.setHorizontalHeaderLabels(
            ["Side", "Entry Idx", "Exit Idx", "Entry Price", "Exit Price", "PnL", "Balance"]
        )
        table.setRowCount(len(trades))
        for r, t in enumerate(trades):
            side, entry_i, exit_i, entry_p, exit_p, pnl, balance = t
            row_vals = [
                side,
                str(entry_i),
                str(exit_i),
                f"{entry_p:.5f}",
                f"{exit_p:.5f}",
                f"{pnl:.2f}",
                f"{balance:.2f}",
            ]
            for c, v in enumerate(row_vals):
                item = QTableWidgetItem(v)
                if c == 5:
                    if pnl > 0:
                        item.setForeground(Qt.darkGreen)
                    elif pnl < 0:
                        item.setForeground(Qt.red)
                table.setItem(r, c, item)
        table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        layout.addWidget(table)

        summary = QLabel()
        currency = "USD" if self.quote_usd.isChecked() else "QUOTE"
        pnl_color = "darkgreen" if total > 0 else ("red" if total < 0 else "black")
        summary.setText(
            f"Trades: {len(trades)} | "
            f"Wins: <span style='color: darkgreen;'>{wins}</span> | "
            f"Losses: <span style='color: red;'>{losses}</span> | "
            f"Total PnL: <span style='color: {pnl_color};'>{total:.2f} {currency}</span>"
        )
        summary.setStyleSheet("font-weight: bold;")
        summary.setTextFormat(Qt.RichText)
        layout.addWidget(summary)

        dlg.exec_()

    def load_mt5_history(self):
        if mt5 is None:
            QMessageBox.critical(self, "MT5 Error", "MetaTrader5 package not available. Install MetaTrader5.")
            return

        symbol = self.symbol.currentText()
        timeframe = self.timeframe.currentText()
        time_from = self.date_from.date().toPyDate()
        time_to = self.date_to.date().toPyDate()

        if time_from > time_to:
            QMessageBox.critical(self, "Date Error", "From date must be <= To date.")
            return

        tf_map = {
            "M5": mt5.TIMEFRAME_M5,
            "M15": mt5.TIMEFRAME_M15,
            "H1": mt5.TIMEFRAME_H1,
            "H2": mt5.TIMEFRAME_H2,
            "H4": mt5.TIMEFRAME_H4,
            "D1": mt5.TIMEFRAME_D1,
        }

        try:
            if not mt5.initialize():
                raise RuntimeError(f"MT5 initialize failed: {mt5.last_error()}")

            info = mt5.symbol_info(symbol)
            if info is None:
                raise ValueError(f"Symbol not found in MT5: {symbol}")
            if not info.visible:
                if not mt5.symbol_select(symbol, True):
                    raise ValueError(f"Symbol not visible/selected: {symbol}")
            if info.trade_contract_size and info.trade_contract_size > 0:
                self.contract_size.setValue(float(info.trade_contract_size))

            dt_from = datetime.combine(time_from, time.min)
            dt_to = datetime.combine(time_to, time.max)
            if dt_from > dt_to:
                raise ValueError("From date must be <= To date.")

            rates = mt5.copy_rates_range(symbol, tf_map[timeframe], dt_from, dt_to)
            if rates is None or len(rates) == 0:
                latest = mt5.copy_rates_from_pos(symbol, tf_map[timeframe], 0, 1)
                if latest is not None and len(latest) > 0:
                    latest_time = pd.to_datetime(latest[0]["time"], unit="s")
                    raise ValueError(
                        f"No data returned for selected range. Latest bar for {symbol} {timeframe}: {latest_time}."
                    )
                raise ValueError("No data returned for selected range.")

            df = pd.DataFrame(rates)
            df["time"] = pd.to_datetime(df["time"], unit="s")
            df = df.rename(columns={"open": "open", "high": "high", "low": "low", "close": "close"})

            if len(df) < 50:
                raise ValueError("Not enough rows. Choose a larger date range.")

            self.df = df.reset_index(drop=True)
            self.btn_run.setEnabled(True)
            self.status.setText(f"Loaded {len(df)} rows from MT5: {symbol} {timeframe} ({time_from} to {time_to})")
            self.redraw(empty=True)

        except Exception as e:
            QMessageBox.critical(self, "Load Error", str(e))
        finally:
            mt5.shutdown()

    def run_rules(self):
        if self.df is None:
            return

        cfg = SMCConfig(
            swing_n=int(self.swing_n.value()),
            choch_lookahead=int(self.choch_k.value()),
            fvg_lookahead=int(self.fvg_k.value()),
            min_gap=float(self.min_gap.value()),
        )

        try:
            self.smc_out = generate_signals(self.df, cfg)
            sig = self.smc_out["signals"]
            buys = int(np.sum(sig == "BUY"))
            sells = int(np.sum(sig == "SELL"))
            self.status.setText(f"Done. Signals: BUY={buys}, SELL={sells}. (Showing last 400 candles)")
            init_bal = float(self.init_balance.value())
            leverage = float(self.leverage.currentText())
            trades, total, wins, losses, gross_profit, gross_loss, end_balance, skipped_entries = backtest_simple(
                self.df,
                sig,
                float(self.lot.value()),
                float(self.contract_size.value()),
                init_bal,
                leverage,
            )
            trade_count = len(trades)
            win_rate = (wins / trade_count * 100.0) if trade_count else 0.0
            avg_pnl = (total / trade_count) if trade_count else 0.0
            profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else float("inf")

            currency = "USD" if self.quote_usd.isChecked() else "QUOTE"
            pf_text = f"{profit_factor:.2f}" if profit_factor != float("inf") else "inf"
            end_bal = end_balance

            self.bt_status.setText(
                f"Backtest Summary ({currency})\n"
                f"Trades: {trade_count} | Wins: {wins} Losses: {losses} | Win Rate: {win_rate:.1f}%\n"
                f"Init Balance: {init_bal:.2f} {currency} | End Balance: {end_bal:.2f} {currency}\n"
                f"Total PnL: {total:.2f} {currency} | Avg/Trade: {avg_pnl:.2f} {currency} | "
                f"Profit Factor: {pf_text} | Skipped Entries: {skipped_entries}"
            )
            self.redraw()

        except Exception as e:
            QMessageBox.critical(self, "Run Error", str(e))

    def redraw(self, empty: bool = False):
        self.canvas.ax.clear()
        if self.df is None or empty:
            self.canvas.ax.set_title("Load a CSV to plot candles")
            self.canvas.draw()
            return

        d = plot_candles(self.canvas.ax, self.df, max_bars=400)
        if self.smc_out is not None:
            plot_markers(self.canvas.ax, d, self.smc_out["signals"])

        self.canvas.draw()


def main():
    app = QApplication(sys.argv)
    w = MainWindow()
    w.resize(1200, 700)
    w.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
