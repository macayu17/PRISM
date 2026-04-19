import { Outlet } from "react-router-dom";
import { motion } from "framer-motion";
import Navbar from "./Navbar";
import LiquidBackground from "./LiquidBackground";
import { Brain } from "lucide-react";
import { pageShell } from "../lib/ui";

const pageTransition = {
  initial: { opacity: 0, y: 20 },
  animate: { opacity: 1, y: 0 },
  exit: { opacity: 0, y: -20 },
  transition: { duration: 0.3, ease: "easeOut" },
};

export default function Layout() {
  return (
    <>
      <LiquidBackground />
      <Navbar />
      <main className="pb-20 pt-24 md:pt-28">
        <motion.div {...pageTransition}>
          <Outlet />
        </motion.div>
      </main>
      <footer className="border-t border-white/10 py-6 text-center text-sm text-slate-500">
        <div className={`${pageShell} flex flex-wrap items-center justify-center gap-2`}>
          <Brain size={16} />
          <span>NeuroAssess — Parkinson&apos;s Disease Assessment System</span>
          <span className="mx-2">•</span>
          <span>Research & Educational Use Only</span>
        </div>
      </footer>
    </>
  );
}
