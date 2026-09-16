import { motion } from "framer-motion";
import BuildStatusSection from "../components/BuildStatusSection";
import { sectionMotion, sectionTransition } from "../pageMotion";

export default function OverviewPage() {
  return (
    <motion.div {...sectionMotion} transition={sectionTransition} className="app-page">
      <BuildStatusSection />
    </motion.div>
  );
}
