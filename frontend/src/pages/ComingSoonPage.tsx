import { motion } from "framer-motion";
import { Link } from "react-router-dom";
import { Button } from "../components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "../components/ui/card";
import { sectionMotion, sectionTransition } from "../pageMotion";

interface ComingSoonPageProps {
  title: string;
  description: string;
}

/**
 * Reused for every PRD research surface not yet built on real backend data
 * (Phase 8B+). Deliberately carries no chart, table, or number of any kind —
 * see PRD.md §10/§11: nothing here may look like a real result before one
 * exists.
 */
export default function ComingSoonPage({ title, description }: ComingSoonPageProps) {
  return (
    <motion.div {...sectionMotion} transition={sectionTransition} className="app-page">
      <Card>
        <CardHeader>
          <CardTitle>{title}</CardTitle>
          <CardDescription>{description}</CardDescription>
        </CardHeader>
        <CardContent className="flex flex-col gap-3 text-sm text-muted-foreground">
          <p>
            This research surface is not yet implemented. RiskFecta only shows real, computed
            results — nothing here is a placeholder chart, mock table, or illustrative figure
            standing in for one.
          </p>
          <div className="flex flex-wrap gap-2 pt-1">
            <Button asChild variant="outline" size="sm">
              <Link to="/methodology">Read the methodology</Link>
            </Button>
            <Button asChild variant="ghost" size="sm">
              <Link to="/">Back to overview</Link>
            </Button>
          </div>
        </CardContent>
      </Card>
    </motion.div>
  );
}
