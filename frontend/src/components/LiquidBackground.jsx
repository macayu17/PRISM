import React from 'react';

export default function LiquidBackground() {
    return (
        <div className="pointer-events-none fixed inset-0 -z-10 overflow-hidden bg-[var(--bg-primary)]">
            <div className="absolute inset-0 bg-[linear-gradient(90deg,rgba(45,212,191,0.045)_1px,transparent_1px),linear-gradient(0deg,rgba(255,255,255,0.035)_1px,transparent_1px)] bg-[size:56px_56px]" />
            <div className="absolute inset-0 bg-[linear-gradient(135deg,rgba(20,184,166,0.10),transparent_36%,rgba(245,158,11,0.07)_72%,transparent)]" />
            <div className="absolute inset-x-0 top-0 h-px bg-gradient-to-r from-transparent via-teal-300/60 to-transparent" />
            <div className="absolute inset-0 bg-[repeating-linear-gradient(0deg,rgba(255,255,255,0.025)_0px,rgba(255,255,255,0.025)_1px,transparent_1px,transparent_5px)] opacity-30" />
        </div>
    );
}
