import { useMemo, useState } from "react";
import type { UniverseTicker } from "../api/types";
import type { AsyncState } from "../hooks/useAsync";
import LoadingState from "./LoadingState";
import ErrorState from "./ErrorState";

interface UniverseBrowserProps {
  state: AsyncState<UniverseTicker[]>;
  selected: string | null;
  onSelect: (ticker: string) => void;
  onRetry: () => void;
}

export default function UniverseBrowser({ state, selected, onSelect, onRetry }: UniverseBrowserProps) {
  const [query, setQuery] = useState("");

  const filtered = useMemo(() => {
    if (state.status !== "success") return [];
    const q = query.trim().toUpperCase();
    if (!q) return state.data;
    return state.data.filter(
      (row) => row.ticker.includes(q) || (row.sector ?? "").toUpperCase().includes(q),
    );
  }, [state, query]);

  return (
    <section className="card" aria-labelledby="universe-heading">
      <div className="universe-header">
        <h2 id="universe-heading">Universe</h2>
        {state.status === "success" && (
          <input
            type="search"
            className="universe-search"
            placeholder="Search ticker or sector…"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            aria-label="Search the ticker universe"
          />
        )}
      </div>

      {state.status === "loading" && <LoadingState label="Loading universe…" />}
      {state.status === "error" && <ErrorState message={state.message} onRetry={onRetry} />}
      {state.status === "success" && (
        <>
          {filtered.length === 0 ? (
            <p className="empty-note">No tickers match "{query}".</p>
          ) : (
            <ul className="ticker-list" role="listbox" aria-label="Ticker universe">
              {filtered.map((row) => (
                <li key={row.ticker}>
                  <button
                    type="button"
                    role="option"
                    aria-selected={row.ticker === selected}
                    className={`ticker-chip${row.ticker === selected ? " ticker-chip--selected" : ""}`}
                    onClick={() => onSelect(row.ticker)}
                  >
                    <span className="ticker-chip__symbol">{row.ticker}</span>
                    {row.sector && <span className="ticker-chip__sector">{row.sector}</span>}
                  </button>
                </li>
              ))}
            </ul>
          )}
        </>
      )}
    </section>
  );
}
