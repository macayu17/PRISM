import React from 'react';

export default function LiquidBackground() {
    return (
        <div className="pointer-events-none fixed inset-0 -z-10 overflow-hidden bg-[var(--bg-primary)]">
            <div className="absolute -left-[10%] -top-[10%] h-[50vw] w-[50vw] animate-orb-1 rounded-full bg-[radial-gradient(circle,_rgba(56,189,248,0.15)_0%,_transparent_70%)] blur-[80px]" />
            <div className="absolute -bottom-[10%] -right-[10%] h-[60vw] w-[60vw] animate-orb-2 rounded-full bg-[radial-gradient(circle,_rgba(129,140,248,0.12)_0%,_transparent_70%)] blur-[100px]" />
            <div className="absolute left-1/2 top-1/2 h-[40vw] w-[40vw] -translate-x-1/2 -translate-y-1/2 animate-orb-pulse rounded-full bg-[radial-gradient(circle,_rgba(255,255,255,0.03)_0%,_transparent_60%)] blur-[60px]" />
        </div>
    );
}
