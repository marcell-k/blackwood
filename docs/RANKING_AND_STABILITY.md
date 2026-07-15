# Ranking Columns and Stability Pipeline

## Column meanings

- `#`: Rank after final composite sorting (`1` is best).
- `Window`: Trading time window (`start_hour:start_min` -> `end_hour:end_min`).
- `RRR`: Risk-reward ratio parameter (`rrr`).
- `Por`: Position portion/size parameter (`portion`).
- `Cnd`: Number of entry/logic conditions (`nmb_c`).
- `IS p25>med`:
  - Left value: CPCV in-sample 25th percentile Sharpe (`cpcv_sharpe_p25`).
  - Bar + right value: CPCV in-sample median Sharpe (`cpcv_sharpe_median`).
- `Stab`: Proximity stability score from neighborhood CV across CPCV folds (`stability_score`).
- `Boot`: Bootstrap stability score from PnL resampling (`bootstrap_stability_score`).
- `OOS Sh`: Holdout Sharpe (`oos_sharpe`).
- `MaxDD%`: Holdout max drawdown percent (`oos_maxdd`).
- `Score`: Final composite score (`composite_score`).

## Stability pipeline (simple)

1. **Phase 1 - CPCV execution**
   - Run optimization across CPCV paths on train data.
   - Build IS metrics (`cpcv_sharpe_p25`, `cpcv_sharpe_median`, fold-level sharpes).

2. **Phase 2 - Dual filter selection**
   - Performance filter: keep stronger IS results.
   - Proximity stability filter: require stable nearby params in normalized param space.
   - Consistency filter: require low CV across folds.
   - Keep top `N` by `phase2_score`.

3. **Phase 3 - Bootstrap validation**
   - Block-bootstrap daily log returns from equity curve.
   - Compute Sharpe/Calmar distribution.
   - Convert Sharpe CV to `bootstrap_stability_score`; apply `pass_bootstrap`.

4. **Phase 4 - OOS validation**
   - Test candidates on holdout split.
   - Compute OOS metrics (`oos_sharpe`, `oos_maxdd`, degradation vs IS).
   - Apply OOS gates (`pass_oos`).

5. **Phase 5 - Ranking and classification**
   - Normalize components (IS performance, IS consistency, proximity stability, bootstrap stability, OOS robustness).
   - Weighted combine into `composite_score`.
   - Sort by score, assign rank and tier.
