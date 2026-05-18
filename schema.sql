-- RiskFecta PostgreSQL Schema (5 tables)
-- PRD v1.0 / Build Plan v1.0 — run_id is logical key only (no FK).
-- Risk-free rate for Sharpe ratio: USGG10YR / 252 (daily). No T-bill.

-- ---------------------------------------------------------------------------
-- Table 1: prices_raw — Bloomberg BQL price export. Immutable after load.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS prices_raw (
    id              SERIAL          PRIMARY KEY,
    ticker          VARCHAR(20)     NOT NULL,
    date            DATE            NOT NULL,
    open            NUMERIC(12,4),
    high            NUMERIC(12,4),
    low             NUMERIC(12,4),
    close           NUMERIC(12,4)   NOT NULL,
    volume          BIGINT,
    total_return_idx NUMERIC(12,4),
    UNIQUE (ticker, date)
);

-- ---------------------------------------------------------------------------
-- Table 2: features — Python-computed indicators + Bloomberg static/macro.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS features (
    id              SERIAL          PRIMARY KEY,
    ticker          VARCHAR(20)     NOT NULL,
    date            DATE            NOT NULL,
    rsi_14          NUMERIC(8,4),
    macd            NUMERIC(10,4),
    macd_signal     NUMERIC(10,4),
    bb_upper        NUMERIC(12,4),
    bb_lower        NUMERIC(12,4),
    volatility_20d  NUMERIC(8,4),
    momentum_3m     NUMERIC(8,4),
    momentum_6m     NUMERIC(8,4),
    beta            NUMERIC(8,4),
    mkt_cap_log     NUMERIC(12,4),
    sector          VARCHAR(50),
    div_yield       NUMERIC(8,4),
    vix             NUMERIC(8,4),
    yield_10y       NUMERIC(8,4),
    UNIQUE (ticker, date)
);

-- ---------------------------------------------------------------------------
-- Table 3: predictions — Ensemble output; actual_return filled post-facto.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS predictions (
    id                  SERIAL          PRIMARY KEY,
    ticker              VARCHAR(20)     NOT NULL,
    forecast_date        DATE            NOT NULL,
    target_date          DATE            NOT NULL,
    lstm_pred           NUMERIC(10,6),
    xgb_pred            NUMERIC(10,6),
    ensemble_pred       NUMERIC(10,6),
    actual_return       NUMERIC(10,6),
    directional_correct BOOLEAN,
    UNIQUE (ticker, forecast_date)
);

-- ---------------------------------------------------------------------------
-- Table 4: portfolios — Optimizer output; weights per Efficient Frontier point.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS portfolios (
    id              SERIAL          PRIMARY KEY,
    run_id          VARCHAR(50)     NOT NULL,
    ticker          VARCHAR(20)     NOT NULL,
    weight          NUMERIC(8,6)    NOT NULL,
    target_return   NUMERIC(8,6),
    portfolio_vol   NUMERIC(8,6),
    sharpe_ratio    NUMERIC(8,4),
    created_at      TIMESTAMP       DEFAULT NOW()
);

-- ---------------------------------------------------------------------------
-- Table 5: risk_metrics — VaR, max drawdown, Sharpe, per-asset contribution.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS risk_metrics (
    id              SERIAL          PRIMARY KEY,
    run_id          VARCHAR(50)     NOT NULL,
    metric_name     VARCHAR(50)     NOT NULL,
    metric_value    NUMERIC(12,6)   NOT NULL,
    ticker          VARCHAR(20),  -- NULL for portfolio-level
    created_at      TIMESTAMP       DEFAULT NOW()
);
