import type { MarketSummaryResponse } from "../api/types";
import type { AsyncState } from "../hooks/useAsync";
import LoadingState from "./LoadingState";
import ErrorState from "./ErrorState";

interface MarketSummarySectionProps {
  state: AsyncState<MarketSummaryResponse>;
  onRetry: () => void;
}

const numberFormat = new Intl.NumberFormat("en-US");

export default function MarketSummarySection({ state, onRetry }: MarketSummarySectionProps) {
  return (
    <section className="card" aria-labelledby="market-summary-heading">
      <h2 id="market-summary-heading">Market data coverage</h2>

      {state.status === "loading" && <LoadingState label="Loading market summary…" />}
      {state.status === "error" && <ErrorState message={state.message} onRetry={onRetry} />}
      {state.status === "success" && (
        <dl className="stat-grid">
          <div className="stat-grid__item">
            <dt>Tickers</dt>
            <dd>{numberFormat.format(state.data.ticker_count)}</dd>
          </div>
          <div className="stat-grid__item">
            <dt>Stored observations</dt>
            <dd>{numberFormat.format(state.data.price_row_count)}</dd>
          </div>
          <div className="stat-grid__item">
            <dt>First available date</dt>
            <dd>{state.data.first_date ?? "—"}</dd>
          </div>
          <div className="stat-grid__item">
            <dt>Last available date</dt>
            <dd>{state.data.last_date ?? "—"}</dd>
          </div>
        </dl>
      )}
    </section>
  );
}
