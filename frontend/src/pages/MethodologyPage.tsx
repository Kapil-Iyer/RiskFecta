import { motion } from "framer-motion";
import MethodologySection from "../components/MethodologySection";
import { sectionMotion, sectionTransition } from "../pageMotion";

export default function MethodologyPage() {
  return (
    <motion.div {...sectionMotion} transition={sectionTransition} className="app-page">
      <MethodologySection />
    </motion.div>
  );
}
