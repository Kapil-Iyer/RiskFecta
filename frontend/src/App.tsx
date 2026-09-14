import { useCallback, useState } from "react";
import { getMarketSummary, getUniverse } from "./api/client";
import { useAsync } from "./hooks/useAsync";
import Header from "./components/Header";
import DataFreshnessBanner from "./components/DataFreshnessBanner";
import BackendStatusBanner from "./components/BackendStatusBanner";
import BuildStatusSection from "./components/BuildStatusSection";
import MarketSummarySection from "./components/MarketSummarySection";
import UniverseBrowser from "./components/UniverseBrowser";
import TickerDetail from "./components/TickerDetail";
import MethodologySection from "./components/MethodologySection";

export default function App() {
  const [summaryAttempt, setSummaryAttempt] = useState(0);
  const [universeAttempt, setUniverseAttempt] = useState(0);
  const [selectedTicker, setSelectedTicker] = useState<string | null>(null);

  const summaryState = useAsync(getMarketSummary, [summaryAttempt]);
  const universeState = useAsync(getUniverse, [universeAttempt]);

  const retrySummary = useCallback(() => setSummaryAttempt((n) => n + 1), []);
  const retryUniverse = useCallback(() => setUniverseAttempt((n) => n + 1), []);

  const lastDate = summaryState.status === "success" ? summaryState.data.last_date : undefined;

  return (
    <div className="app-shell">
      <Header />
      <DataFreshnessBanner lastDate={lastDate} />
      <BackendStatusBanner />

      <main className="app-main">
        <BuildStatusSection />
        <MarketSummarySection state={summaryState} onRetry={retrySummary} />

        <div className="universe-layout">
          <UniverseBrowser
            state={universeState}
            selected={selectedTicker}
            onSelect={setSelectedTicker}
            onRetry={retryUniverse}
          />
          {selectedTicker ? (
            <TickerDetail ticker={selectedTicker} />
          ) : (
            <section className="card ticker-detail ticker-detail--empty">
              <p className="empty-note">Select a ticker from the universe to view its historical price chart.</p>
            </section>
          )}
        </div>

        <MethodologySection />
      </main>

      <footer className="app-footer">
        <p>RiskFecta — academic / research project. Not investment advice.</p>
      </footer>
    </div>
  );
}
