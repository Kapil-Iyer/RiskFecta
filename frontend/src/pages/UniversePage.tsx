import { useCallback, useState } from "react";
import { motion } from "framer-motion";
import { useOutletContext } from "react-router-dom";
import type { AppOutletContext } from "../AppShell";
import { getUniverse } from "../api/client";
import { useAsync } from "../hooks/useAsync";
import MarketSummarySection from "../components/MarketSummarySection";
import UniverseBrowser from "../components/UniverseBrowser";
import TickerDetail from "../components/TickerDetail";
import { sectionMotion, sectionTransition } from "../pageMotion";

export default function UniversePage() {
  const { summaryState, retrySummary } = useOutletContext<AppOutletContext>();
  const [universeAttempt, setUniverseAttempt] = useState(0);
  const [selectedTicker, setSelectedTicker] = useState<string | null>(null);

  const universeState = useAsync(getUniverse, [universeAttempt]);
  const retryUniverse = useCallback(() => setUniverseAttempt((n) => n + 1), []);

  return (
    <motion.div {...sectionMotion} transition={sectionTransition} className="app-page">
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
    </motion.div>
  );
}
