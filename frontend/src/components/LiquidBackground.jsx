import React from 'react';

export default function LiquidBackground() {
    return (
        <div style={{
            position: 'fixed',
            top: 0,
            left: 0,
            width: '100vw',
            height: '100vh',
            zIndex: -1,
            overflow: 'hidden',
            background: 'var(--bg-primary)',
        }}>
            {/* Orb 1 - Sky Blue */}
            <div style={{
                position: 'absolute',
                top: '-10%',
                left: '-10%',
                width: '50vw',
                height: '50vw',
                background: 'radial-gradient(circle, rgba(56, 189, 248, 0.15) 0%, transparent 70%)',
                filter: 'blur(80px)',
                borderRadius: '50%',
                animation: 'moveOrb1 20s infinite alternate ease-in-out',
            }} />

            {/* Orb 2 - Indigo */}
            <div style={{
                position: 'absolute',
                bottom: '-10%',
                right: '-10%',
                width: '60vw',
                height: '60vw',
                background: 'radial-gradient(circle, rgba(129, 140, 248, 0.12) 0%, transparent 70%)',
                filter: 'blur(100px)',
                borderRadius: '50%',
                animation: 'moveOrb2 25s infinite alternate-reverse ease-in-out',
            }} />

            {/* Orb 3 - Subtle Center Glow */}
            <div style={{
                position: 'absolute',
                top: '50%',
                left: '50%',
                transform: 'translate(-50%, -50%)',
                width: '40vw',
                height: '40vw',
                background: 'radial-gradient(circle, rgba(255, 255, 255, 0.03) 0%, transparent 60%)',
                filter: 'blur(60px)',
                borderRadius: '50%',
                animation: 'pulseOrb 15s infinite ease-in-out',
            }} />

            <style>{`
        @keyframes moveOrb1 {
          0% { transform: translate(0, 0); }
          100% { transform: translate(10%, 10%); }
        }
        @keyframes moveOrb2 {
          0% { transform: translate(0, 0); }
          100% { transform: translate(-10%, -10%); }
        }
        @keyframes pulseOrb {
          0%, 100% { opacity: 0.5; transform: translate(-50%, -50%) scale(1); }
          50% { opacity: 0.8; transform: translate(-50%, -50%) scale(1.1); }
        }
      `}</style>
        </div>
    );
}
