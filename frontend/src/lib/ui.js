/* ── shared UI class-name tokens ─────────────────────────────────── */

const buttonBase =
  "inline-flex items-center justify-center gap-2 rounded-md px-5 py-3 text-sm font-semibold transition duration-300 cursor-pointer disabled:cursor-not-allowed disabled:opacity-60";

/* ── page shells (narrower + generous padding) ───────────────────── */
export const pageShell = "mx-auto w-full max-w-6xl px-5 sm:px-8 lg:px-10";
export const pageShellWide = "mx-auto w-full max-w-7xl px-5 sm:px-8 lg:px-10";
export const pageShellNarrow = "mx-auto w-full max-w-5xl px-5 sm:px-8 lg:px-10";

/* ── glass panels ────────────────────────────────────────────────── */
export const glassPanel =
  "relative overflow-hidden rounded-lg border border-white/[0.10] bg-[rgba(10,17,16,0.72)] p-6 shadow-[0_2px_12px_rgba(0,0,0,0.15)] backdrop-blur-xl sm:p-8";

export const glassPanelInteractive = `${glassPanel} transition duration-300 ease-out hover:-translate-y-0.5 hover:border-teal-300/25 hover:bg-white/[0.05] hover:shadow-[0_8px_24px_rgba(0,0,0,0.18)]`;

export const innerPanel =
  "rounded-md border border-white/[0.08] bg-black/30 p-5";

/* ── typography helpers ──────────────────────────────────────────── */
export const sectionTitle =
  "mb-3 block text-[11px] font-semibold uppercase tracking-normal text-teal-300";

export const sectionHeading =
  "mb-6 flex items-center gap-3 border-b border-white/[0.08] pb-3 text-white";

/* ── form elements ───────────────────────────────────────────────── */
export const inputField =
  "w-full rounded-md border border-white/[0.1] bg-black/30 px-4 py-3 text-sm text-white outline-none transition duration-200 placeholder:text-slate-500 focus:border-teal-300/60 focus:bg-black/40 focus:ring-1 focus:ring-teal-300/20";

export const selectField =
  "w-full rounded-md border border-white/[0.1] bg-black/30 px-4 py-3 pr-10 text-sm text-white outline-none transition duration-200 appearance-none cursor-pointer bg-[url('data:image/svg+xml;charset=utf-8,%3Csvg%20xmlns%3D%22http%3A%2F%2Fwww.w3.org%2F2000%2Fsvg%22%20width%3D%2216%22%20height%3D%2216%22%20fill%3D%22%2394a3b8%22%20viewBox%3D%220%200%2024%2024%22%3E%3Cpath%20d%3D%22M7%2010l5%205%205-5z%22%2F%3E%3C%2Fsvg%3E')] bg-[length:20px] bg-[right_12px_center] bg-no-repeat focus:border-teal-300/60 focus:bg-black/40 focus:ring-1 focus:ring-teal-300/20";

export const labelText =
  "mb-2 flex items-center gap-1.5 text-sm font-medium tracking-wide text-slate-400";

/* ── buttons ─────────────────────────────────────────────────────── */
export const buttonPrimary = `${buttonBase} bg-teal-200 text-black shadow-[0_0_12px_rgba(45,212,191,0.12)] hover:bg-teal-100 hover:shadow-[0_0_20px_rgba(45,212,191,0.20)] active:scale-[0.98]`;
export const buttonSecondary = `${buttonBase} border border-white/10 bg-white/[0.03] text-white hover:border-white/20 hover:bg-white/[0.07]`;
export const buttonSuccess = `${buttonBase} border border-emerald-400/20 bg-emerald-400/10 px-4 py-2 text-emerald-300 hover:bg-emerald-400/20`;
export const buttonDanger = `${buttonBase} border border-rose-400/25 bg-rose-400/10 px-4 py-2 text-rose-200 hover:bg-rose-400/20`;

/* ── badges ──────────────────────────────────────────────────────── */
const badgeBase =
  "inline-flex items-center rounded-sm border px-2.5 py-1 text-[10px] leading-none font-semibold uppercase tracking-normal";

const badgeTones = {
  success: "border-emerald-400/20 bg-emerald-400/10 text-emerald-300",
  warning: "border-amber-400/20 bg-amber-400/10 text-amber-300",
  danger: "border-rose-400/20 bg-rose-400/10 text-rose-300",
  info: "border-sky-400/20 bg-sky-400/10 text-sky-200",
  accent: "border-indigo-400/20 bg-indigo-400/10 text-indigo-200",
};

export function badgeClass(tone = "info") {
  return `${badgeBase} ${badgeTones[tone] ?? badgeTones.info}`;
}

/* ── alerts ──────────────────────────────────────────────────────── */
const alertBase =
  "mb-5 flex items-start gap-3 rounded-md border px-4 py-3 text-sm";

const alertTones = {
  danger: "border-rose-400/20 bg-rose-400/10 text-rose-100",
  success: "border-emerald-400/20 bg-emerald-400/10 text-emerald-100",
  info: "border-sky-400/20 bg-sky-400/10 text-sky-100",
  warning: "border-amber-400/20 bg-amber-400/10 text-amber-50",
};

export function alertClass(tone = "info") {
  return `${alertBase} ${alertTones[tone] ?? alertTones.info}`;
}

/* ── misc ────────────────────────────────────────────────────────── */
export const progressTrack = "h-2 w-full overflow-hidden rounded-sm bg-white/10";

export const reportShell =
  "max-h-[500px] overflow-y-auto rounded-md border border-white/[0.08] bg-black/40 p-5 font-mono text-sm leading-7 whitespace-pre-wrap break-words text-slate-300";
