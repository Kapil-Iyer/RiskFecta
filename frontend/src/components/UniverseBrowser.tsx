import { useMemo, useState } from "react";
import { motion } from "framer-motion";
import { Search } from "lucide-react";
import type { UniverseTicker } from "../api/types";
import type { AsyncState } from "../hooks/useAsync";
import { companyName } from "../data/companyNames";
import { sectorForTicker } from "../data/tickerUniverse";
import { sectorAttr, themeForSector } from "../theme";
import LoadingState from "./LoadingState";
import ErrorState from "./ErrorState";
import Skeleton from "./Skeleton";

interface UniverseBrowserProps {
  state: AsyncState<UniverseTicker[]>;
  selected: string | null;
  onSelect: (ticker: string) => void;
  onRetry: () => void;
}

function matches(row: UniverseTicker, query: string): boolean {
  const q = query.trim().toLowerCase();
  if (!q) return true;
  const sector = row.sector ?? sectorForTicker(row.ticker) ?? "";
  return (
    row.ticker.toLowerCase().includes(q) ||
    companyName(row.ticker).toLowerCase().includes(q) ||
    sector.toLowerCase().includes(q)
  );
}

export default function UniverseBrowser({ state, selected, onSelect, onRetry }: UniverseBrowserProps) {
  const [query, setQuery] = useState("");

  const filtered = useMemo(() => {
    if (state.status !== "success") return [];
    return state.data.filter((row) => matches(row, query));
  }, [state, query]);

  return (
    <section className="card" aria-labelledby="universe-heading">
      <div className="universe-header">
        <h2 id="universe-heading">Universe</h2>
        {state.status === "success" && (
          <div className="universe-search-field">
            <Search size={15} aria-hidden="true" className="universe-search-icon" />
            <input
              type="search"
              className="universe-search"
              placeholder="Search ticker, company, or sector…"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              aria-label="Search the ticker universe by ticker, company name, or sector"
            />
          </div>
        )}
      </div>

      {state.status === "loading" && (
        <>
          <LoadingState label="Loading universe…" />
          <div className="ticker-list" aria-hidden="true">
            {Array.from({ length: 6 }).map((_, i) => (
              <Skeleton key={i} height="2.75rem" />
            ))}
          </div>
        </>
      )}
      {state.status === "error" && <ErrorState message={state.message} onRetry={onRetry} />}
      {state.status === "success" && (
        <>
          {filtered.length === 0 ? (
            <p className="empty-note">No matches for "{query}".</p>
          ) : (
            <ul className="ticker-list" role="listbox" aria-label="Ticker universe">
              {filtered.map((row) => {
                const sector = row.sector ?? sectorForTicker(row.ticker);
                const theme = themeForSector(sector);
                const isSelected = row.ticker === selected;
                return (
                  <motion.li key={row.ticker} layout transition={{ duration: 0.18 }}>
                    <button
                      type="button"
                      role="option"
                      aria-selected={isSelected}
                      className={`ticker-chip${isSelected ? " ticker-chip--selected" : ""}`}
                      data-sector={sectorAttr(sector)}
                      style={{ ["--local-accent" as string]: theme.cssVar }}
                      onClick={() => onSelect(row.ticker)}
                    >
                      <span className="ticker-chip__identity">
                        <span className="ticker-chip__symbol">{row.ticker}</span>
                        <span className="ticker-chip__name">{companyName(row.ticker)}</span>
                      </span>
                      {sector && <span className="ticker-chip__sector">{sector}</span>}
                    </button>
                  </motion.li>
                );
              })}
            </ul>
          )}
        </>
      )}
    </section>
  );
}
