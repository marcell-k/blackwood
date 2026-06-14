import numpy as np
import pandas as pd
from src.config import RANDOM_STATE


class OHLCBootstrap:
    """
    Unified bootstrap framework for OHLC time series robustness testing.
    Implements permutation and stationary block bootstrap methods for
    Monte Carlo walk-forward validation of trading strategies.
    """

    def __init__(
        self, method: str = "stationary_block", avg_block_length: int | None = None, seed: int | None = RANDOM_STATE
    ):
        self.method = method.lower()
        self.avg_block_length = avg_block_length
        self._rng = np.random.default_rng(seed)

    def generate(self, ohlc: pd.DataFrame, train_window: int | None = None) -> pd.DataFrame:
        time_index = ohlc.index
        n_bars = len(time_index)
        perm_n = n_bars if train_window is None else min(train_window, n_bars)
        if train_window and train_window > n_bars:
            raise ValueError(f"train_window ({train_window}) exceeds data length ({n_bars})")

        ohlc_array = ohlc[["Open", "High", "Low", "Close"]].to_numpy()
        log_ohlc = np.log(ohlc_array)
        log_train = log_ohlc[:perm_n, :]
        train_index = time_index[:perm_n]
        start_log = log_train[0, :]

        (rel_open, rel_h, rel_l, rel_c, overnight_pos, intraday_pos) = self._compute_relatives(
            log_train, train_index, perm_n
        )

        if self.method == "permutation":
            synthetic_log = self._generate_permutation(
                start_log, rel_open, rel_h, rel_l, rel_c, overnight_pos, intraday_pos, perm_n
            )
        else:
            synthetic_log = self._generate_stationary_block(
                start_log, rel_open, rel_h, rel_l, rel_c, overnight_pos, intraday_pos, perm_n, log_train
            )

        full_log = synthetic_log if perm_n == n_bars else np.concatenate((synthetic_log, log_ohlc[perm_n:, :]), axis=0)
        prices = np.exp(full_log)
        return self._build_dataframe(prices, time_index, ohlc)

    def _compute_relatives(self, log_train: np.ndarray, train_index: pd.Index, perm_n: int) -> tuple:
        log_o, log_h, log_l, log_c = log_train[:, 0], log_train[:, 1], log_train[:, 2], log_train[:, 3]

        rel_h = log_h - log_o
        rel_l = log_l - log_o
        rel_c = log_c - log_o

        rel_open = np.zeros_like(log_o)
        if perm_n > 1:
            rel_open[1:] = log_o[1:] - log_c[:-1]

        dates = train_index.date
        day_changes = np.zeros(perm_n, dtype=bool)
        if perm_n > 1:
            day_changes[1:] = np.array(dates[1:]) != np.array(dates[:-1])
        overnight_pos = np.nonzero(day_changes)[0]
        intraday_pos = np.nonzero(~day_changes)[0]

        if len(intraday_pos) > 0:
            rel_open[intraday_pos] = 0.0

        return rel_open, rel_h, rel_l, rel_c, overnight_pos, intraday_pos

    def _reconstruct_ohlc(
        self,
        start_log: np.ndarray,
        r_open: np.ndarray,
        r_h: np.ndarray,
        r_l: np.ndarray,
        r_c: np.ndarray,
        perm_n: int,
        enforce: bool = False,
    ) -> np.ndarray:
        delta = r_open + r_c

        log_close = np.zeros_like(r_open)
        log_close[0] = start_log[3]
        if perm_n > 1:
            log_close[1:] = start_log[3] + np.cumsum(delta[1:])

        log_open = np.zeros_like(log_close)
        log_open[0] = start_log[0]
        if perm_n > 1:
            log_open[1:] = log_close[:-1] + r_open[1:]

        log_high = log_open + r_h
        log_low = log_open + r_l

        if enforce:
            max_oc = np.maximum(log_open, log_close)
            min_oc = np.minimum(log_open, log_close)
            log_high = np.maximum(log_high, max_oc)
            log_low = np.minimum(log_low, min_oc)

        log_ohlc = np.stack((log_open, log_high, log_low, log_close), axis=-1)
        return log_ohlc

    def _generate_permutation(self, start_log, rel_open, rel_h, rel_l, rel_c, overnight_pos, intraday_pos, perm_n):
        if perm_n > 1:
            perm = self._rng.permutation(perm_n)
            rel_h = rel_h[perm]
            rel_l = rel_l[perm]
            rel_c = rel_c[perm]

        if len(overnight_pos) > 1:
            perm = self._rng.permutation(len(overnight_pos))
            overnight_gaps = rel_open[overnight_pos]
            rel_open = rel_open.copy()
            rel_open[overnight_pos] = overnight_gaps[perm]

        synthetic_log = self._reconstruct_ohlc(start_log, rel_open, rel_h, rel_l, rel_c, perm_n, enforce=True)
        return synthetic_log

    def _generate_stationary_block(
        self, start_log, rel_open, rel_h, rel_l, rel_c, overnight_pos, intraday_pos, perm_n, log_train
    ):
        if self.avg_block_length is None:
            log_c = log_train[:, 3]
            if perm_n > 1:
                returns = np.diff(log_c)
                block_len = int(self._get_optimal_length(returns))
            else:
                block_len = 5
        else:
            block_len = self.avg_block_length

        all_relatives = np.stack((rel_open, rel_h, rel_l, rel_c), axis=-1)

        # Bootstrap from all bars with non-circular sampling to preserve return distribution
        boot_relatives = self._stationary_block_resample(all_relatives, perm_n, block_len)

        boot_r_open = boot_relatives[:, 0]
        boot_r_h = boot_relatives[:, 1]
        boot_r_l = boot_relatives[:, 2]
        boot_r_c = boot_relatives[:, 3]

        # Ensure bar 0 has zero gap (no previous bar)
        boot_r_open[0] = 0.0

        return self._reconstruct_ohlc(start_log, boot_r_open, boot_r_h, boot_r_l, boot_r_c, perm_n, enforce=True)

    def _stationary_block_resample(self, data: np.ndarray, n_samples: int, avg_block_length: int) -> np.ndarray:
        n_obs, n_features = data.shape

        if avg_block_length >= n_obs:
            indices = self._rng.integers(0, n_obs, size=n_samples)
            return data[indices]

        p = 1.0 / avg_block_length
        resampled = np.empty((n_samples, n_features), dtype=data.dtype)
        idx = 0

        while idx < n_samples:
            start = self._rng.integers(0, n_obs)
            block_length = self._rng.geometric(p)

            # Non-circular: truncate blocks at data boundary
            end = min(start + block_length, n_obs)
            actual_length = end - start

            if idx + actual_length > n_samples:
                actual_length = n_samples - idx

            resampled[idx : idx + actual_length] = data[start : start + actual_length]
            idx += actual_length

        return resampled

    def _build_dataframe(self, prices: np.ndarray, time_index, orig: pd.DataFrame):
        df = pd.DataFrame(prices, index=time_index, columns=["Open", "High", "Low", "Close"])
        for col in orig.columns:
            if col not in {"Open", "High", "Low", "Close"}:
                df[col] = orig[col].values
        return df

    def _get_optimal_length(self, data: np.ndarray) -> float:
        n = data.shape[0]
        kn = max(5, np.sqrt(np.log10(n)))
        mmax = int(np.ceil(np.sqrt(n)) + kn)
        bmax = np.ceil(min(3 * np.sqrt(n), n / 3))
        c = 2

        temp = self._mlag(data, mmax)
        temp = temp[mmax:]
        corcoef = np.zeros(mmax)
        for iCor in range(mmax):
            corcoef[iCor] = np.corrcoef(data[mmax:], temp[:, iCor])[0, 1]

        temp2 = self._mlag(corcoef, kn).T
        temp3 = np.zeros((kn, corcoef.shape[0] + 1 - kn))

        for iRow in range(kn):
            temp3[iRow, :] = np.append(temp2[iRow, kn : corcoef.shape[0]], corcoef[len(corcoef) - kn + iRow])

        threshold = np.abs(temp3) < (c * np.sqrt(np.log10(n) / n))
        threshold = np.sum(threshold, axis=0)

        mhat = None
        for count, x in enumerate(threshold):
            if x == kn:
                mhat = count
                break

        if mhat is None:
            seccrit = corcoef > (c * np.sqrt(np.log10(n) / n))
            for iLag in range(seccrit.shape[0] - 1, -1, -1):
                if seccrit[iLag]:
                    mhat = iLag + 1
                    break

        if mhat is None:
            M = 0
        elif 2 * mhat > mmax:
            M = mmax
        else:
            M = 2 * mhat

        kk = np.arange(-M, M + 1, 1)

        if M > 0:
            temp = self._mlag(data, M)
            temp = temp[M:]
            temp2 = np.zeros((temp.shape[0], temp.shape[1] + 1))
            for iRow in range(len(data) - M):
                temp2[iRow, :] = np.hstack((data[M + iRow], temp[iRow, :]))

            temp2 = temp2.T
            temp3 = np.cov(temp2)
            acv = temp3[:, 0]

            acv2 = np.zeros((len(acv) - 1, 2))
            acv2[:, 0] = -np.linspace(1, M, M)
            acv2[:, 1] = acv[1 : len(acv)]

            if acv2.shape[0] > 1:
                acv2 = acv2[::-1]

            acv3 = np.concatenate((acv2[:, 1], acv))

            Ghat = 0
            DSBhat = 0
            LamTemp = self._lam(kk / M)

            for iHat in range(acv3.shape[0]):
                Ghat += LamTemp[iHat] * np.abs(kk[iHat]) * acv3[iHat]
                DSBhat += LamTemp[iHat] * acv3[iHat]
            DSBhat = 2 * np.square(DSBhat)

            Bstar = np.power(2 * np.square(Ghat) / DSBhat, 1 / 3) * np.power(n, 1 / 3)

            if Bstar > bmax:
                Bstar = bmax
        else:
            Bstar = 1
        return Bstar

    @staticmethod
    def _mlag(x: np.ndarray, n: int) -> np.ndarray:
        nobs = x.shape[0]
        out = np.zeros((nobs, n))
        for iLag in range(1, n + 1):
            out[iLag:, iLag - 1] = x[: nobs - iLag]
        return out

    @staticmethod
    def _lam(x: np.ndarray) -> np.ndarray:
        abs_x = np.abs(x)
        out = (abs_x < 0.5).astype(float) + 2 * (1 - abs_x) * (abs_x >= 0.5) * (abs_x <= 1)
        return out
