import { useCallback, useState } from "react";
import { getPrices } from "../api/client";
import type { PriceHistoryResponse } from "../api/types";
import { useAsync } from "../hooks/useAsync";
import DateRangeFilter from "./DateRangeFilter";
import LoadingState from "./LoadingState";
import ErrorState from "./ErrorState";
import PriceChart from "./PriceChart";
import VolumeChart from "./VolumeChart";

interface TickerDetailProps {
  ticker: string;
}

export default function TickerDetail({ ticker }: TickerDetailProps) {
  const [start, setStart] = useState("");
  const [end, setEnd] = useState("");
  const [attempt, setAttempt] = useState(0);

  const invalidRange = Boolean(start && end && start > end);

  const state = useAsync<PriceHistoryResponse | null>(() => {
    if (invalidRange) return Promise.resolve(null);
    return getPrices(ticker, { start: start || undefined, end: end || undefined });
  }, [ticker, start, end, invalidRange, attempt]);

  const retry = useCallback(() => setAttempt((n) => n + 1), []);
  const clearRange = useCallback(() => {
    setStart("");
    setEnd("");
  }, []);

  return (
    <section className="card ticker-detail" aria-labelledby="ticker-detail-heading">
      <div className="ticker-detail__header">
        <h2 id="ticker-detail-heading">{ticker}</h2>
        {state.status === "success" && state.data && (
          <p className="ticker-detail__range">
            {state.data.prices.length > 0
              ? `${state.data.prices[0].date} → ${state.data.prices[state.data.prices.length - 1].date} (${state.data.count} sessions)`
              : "No observations in the selected range"}
          </p>
        )}
      </div>

      <DateRangeFilter
        start={start}
        end={end}
        onStartChange={setStart}
        onEndChange={setEnd}
        onClear={clearRange}
        invalid={invalidRange}
      />

      {!invalidRange && state.status === "loading" && <LoadingState label={`Loading ${ticker} price history…`} />}
      {!invalidRange && state.status === "error" && <ErrorState message={state.message} onRetry={retry} />}
      {!invalidRange && state.status === "success" && state.data && state.data.prices.length === 0 && (
        <p className="empty-note">No price observations found for this date range.</p>
      )}
      {!invalidRange && state.status === "success" && state.data && state.data.prices.length > 0 && (
        <div className="ticker-detail__charts">
          <PriceChart ticker={ticker} prices={state.data.prices} />
          <VolumeChart ticker={ticker} prices={state.data.prices} />
        </div>
      )}
    </section>
  );
}
