import { useCallback, useEffect, useState } from "react";
import { motion } from "framer-motion";
import { getPrices } from "../api/client";
import type { PriceHistoryResponse } from "../api/types";
import { useAsync } from "../hooks/useAsync";
import { companyName } from "../data/companyNames";
import { sectorForTicker } from "../data/tickerUniverse";
import { sectorAttr, themeForSector } from "../theme";
import { computeQuickRange, formatLongDate, formatMonthYear, QUICK_RANGES, type QuickRangeKey } from "../dateRanges";
import DateRangeFilter from "./DateRangeFilter";
import LoadingState from "./LoadingState";
import ErrorState from "./ErrorState";
import Badge from "./Badge";
import Skeleton from "./Skeleton";
import PriceChart from "./PriceChart";
import VolumeChart from "./VolumeChart";

interface TickerDetailProps {
  ticker: string;
}

export default function TickerDetail({ ticker }: TickerDetailProps) {
  const [start, setStart] = useState("");
  const [end, setEnd] = useState("");
  const [fullAttempt, setFullAttempt] = useState(0);
  const [filteredAttempt, setFilteredAttempt] = useState(0);

  // Reset any date filter when the selected ticker changes — a fresh view
  // per ticker, not a leftover narrow range from the previous one.
  useEffect(() => {
    setStart("");
    setEnd("");
  }, [ticker]);

  const invalidRange = Boolean(start && end && start > end);
  const hasFilter = Boolean(start || end) && !invalidRange;

  // Always-unfiltered fetch: the source for the stable header stats (latest
  // observation, total sessions, full date range) — and, when no filter is
  // active, doubles as the chart data too, avoiding a redundant duplicate
  // request for the common "no filter yet" case.
  const fullState = useAsync(() => getPrices(ticker), [ticker, fullAttempt]);

  const filteredState = useAsync<PriceHistoryResponse | null>(() => {
    if (!hasFilter) return Promise.resolve(null);
    return getPrices(ticker, { start: start || undefined, end: end || undefined });
  }, [ticker, start, end, hasFilter, filteredAttempt]);

  const chartState = hasFilter ? filteredState : fullState;

  const retryFull = useCallback(() => setFullAttempt((n) => n + 1), []);
  const retryFiltered = useCallback(() => setFilteredAttempt((n) => n + 1), []);
  const clearRange = useCallback(() => {
    setStart("");
    setEnd("");
  }, []);

  const applyQuickRange = useCallback(
    (key: QuickRangeKey) => {
      if (fullState.status !== "success" || fullState.data.prices.length === 0) return;
      const latest = fullState.data.prices[fullState.data.prices.length - 1].date;
      const range = computeQuickRange(latest, key);
      if (range) {
        setStart(range.start);
        setEnd(range.end);
      } else {
        setStart("");
        setEnd("");
      }
    },
    [fullState],
  );

  const sector = sectorForTicker(ticker);
  const theme = themeForSector(sector);
  const name = companyName(ticker);

  return (
    <motion.section
      key={ticker}
      layout
      initial={{ opacity: 0, y: 6 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.25 }}
      className="card ticker-detail"
      data-sector={sectorAttr(sector)}
      style={{ ["--local-accent" as string]: theme.cssVar }}
      aria-labelledby="ticker-detail-heading"
    >
      <div className="ticker-detail__header">
        <div>
          <h2 id="ticker-detail-heading">{name}</h2>
          {sector && <p className="ticker-detail__sector">{sector}</p>}
        </div>
        <div className="ticker-detail__identity">
          <span className="ticker-detail__symbol">{ticker}</span>
          <Badge tone="neutral">Historical data</Badge>
        </div>
      </div>

      {fullState.status === "loading" && (
        <div className="ticker-detail__stats">
          <Skeleton height="2.5rem" width="10rem" />
          <Skeleton height="2.5rem" width="14rem" />
        </div>
      )}
      {fullState.status === "error" && <ErrorState message={fullState.message} onRetry={retryFull} />}
      {fullState.status === "success" && fullState.data.prices.length > 0 && (
        <div className="ticker-detail__stats">
          <div className="stat-block">
            <p className="stat-block__label">Latest dataset observation</p>
            <p className="stat-block__value">${fullState.data.prices[fullState.data.prices.length - 1].close.toFixed(2)}</p>
            <p className="stat-block__meta">{formatLongDate(fullState.data.prices[fullState.data.prices.length - 1].date)}</p>
          </div>
          <div className="stat-block">
            <p className="stat-block__label">Trading sessions</p>
            <p className="stat-block__value">{fullState.data.count.toLocaleString("en-US")}</p>
            <p className="stat-block__meta">
              {formatMonthYear(fullState.data.prices[0].date)} — {formatMonthYear(fullState.data.prices[fullState.data.prices.length - 1].date)}
            </p>
          </div>
        </div>
      )}

      <div className="quick-ranges" role="group" aria-label="Quick date ranges">
        {QUICK_RANGES.map((key) => (
          <button
            key={key}
            type="button"
            className="quick-range-button"
            disabled={fullState.status !== "success"}
            onClick={() => applyQuickRange(key)}
          >
            {key}
          </button>
        ))}
      </div>

      <DateRangeFilter
        start={start}
        end={end}
        onStartChange={setStart}
        onEndChange={setEnd}
        onClear={clearRange}
        invalid={invalidRange}
      />

      {/* Plain conditional rendering, not AnimatePresence: an exit/enter
          choreography here previously left the UI stuck on "loading" when a
          quick-range click fired a new fetch before the prior exit animation
          settled — a real regression, exactly the "animation that delays
          access to information" this project explicitly avoids. Each
          success render still gets a subtle one-way fade via `key` + `initial`
          (mount-only, no exit to coordinate). */}
      {!invalidRange && chartState.status === "loading" && (
        <LoadingState label={`Loading ${ticker} price history…`} />
      )}
      {!invalidRange && chartState.status === "error" && (
        <ErrorState message={chartState.message} onRetry={hasFilter ? retryFiltered : retryFull} />
      )}
      {!invalidRange && chartState.status === "success" && chartState.data && chartState.data.prices.length === 0 && (
        <p className="empty-note">No price observations found for this date range.</p>
      )}
      {!invalidRange && chartState.status === "success" && chartState.data && chartState.data.prices.length > 0 && (
        <motion.div
          key={`charts-${start}-${end}`}
          className="ticker-detail__charts"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.2 }}
        >
          <PriceChart ticker={ticker} prices={chartState.data.prices} accentColor={theme.hex} />
          <VolumeChart ticker={ticker} prices={chartState.data.prices} accentColor={theme.hex} />
        </motion.div>
      )}
    </motion.section>
  );
}
