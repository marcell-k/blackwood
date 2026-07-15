import numpy as np
from backtesting import Strategy


class BaseTemplateStrategy(Strategy):
    fixed_risk_pct = 0.005  # 0.5% fixed risk per trade
    max_risk_pct = 0.02  # 2% maximum risk limit

    # === RISK MANAGEMENT PARAMETERS ===
    drawdown_scale_threshold = 0.05  # Scale down at 5% drawdown
    drawdown_scale_factor = 0.5  # Reduce size by 50%

    def __init__(self) -> None:
        """Initialize strategy - override this method in subclass"""
        super().init()

        self._peak_equity = float(self.equity)  # Explicit copy, not reference
        self._partial_trades: set[int] = set()

    def _get_current_drawdown(self) -> float:
        """Calculate current drawdown with proper peak tracking."""
        if self.equity > self._peak_equity:
            self._peak_equity = float(self.equity)

        if self._peak_equity > 0:
            drawdown = (self._peak_equity - self.equity) / self._peak_equity
            return max(0.0, drawdown)
        else:
            return 0.0

    def _calculate_size(self, sl: float) -> int:
        """Calculate total position size for all 3 positions combined."""
        base_risk_amount = self.equity * 0.01
        max_risk = self.equity * 0.01
        # current_drawdown = self._get_current_drawdown()
        risk_amount = min(base_risk_amount, max_risk)

        # if current_drawdown > 0.05:
        #     risk_amount = risk_amount / 2
        # if current_drawdown > 0.075:
        #     risk_amount = risk_amount / 2

        risk_amount = int(risk_amount / sl)
        return risk_amount

    def next(self) -> None:
        pass


class MetaLabeling(BaseTemplateStrategy):
    gate_col = None
    bet_col = None
    gate_min_long = 0.0
    gate_min_short = 0.0

    use_kelly_sizing = False

    def __init__(self) -> None:
        """Initialize strategy - override this method in subclass"""
        super().init()

        self._peak_equity = float(self.equity)

        if self.gate_col is not None:
            self._gate = self.I(lambda: self.data.df[self.gate_col].values.astype(float), plot=False)
        if self.bet_col is not None:
            self._bet = self.I(lambda: self.data.df[self.bet_col].values.astype(float), plot=False)

    def _gate_allows_long(self) -> bool:
        if self.gate_col is None:
            return True
        val = self._gate[-1]
        return np.isfinite(val) and (val >= self.gate_min_long)

    def _gate_allows_short(self) -> bool:
        if self.gate_col is None:
            return True
        val = self._gate[-1]
        return np.isfinite(val) and (val >= self.gate_min_short)

    def next(self) -> None:
        pass


class BuyAndHoldStrategy(Strategy):
    def __init__(self) -> None:
        """Initialize strategy - no indicators needed for Buy & Hold"""
        pass

    def next(self) -> None:
        """Execute strategy logic on each bar"""
        # Enter long position on first opportunity if not already positioned
        if not self.position:
            self.buy()  # Buy with all available capital

        # Hold position - no selling logic needed
