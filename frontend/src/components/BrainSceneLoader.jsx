import { lazy, Suspense } from 'react';

const BrainScene = lazy(() => import('./BrainScene'));

export default function BrainSceneLoader({ symptomData }) {
  return (
    <Suspense
      fallback={
        <div className="flex h-full w-full items-center justify-center bg-[linear-gradient(135deg,rgba(45,212,191,0.08),rgba(245,158,11,0.05))] text-xs font-semibold uppercase text-slate-400">
          Loading neural model...
        </div>
      }
    >
      <BrainScene symptomData={symptomData} />
    </Suspense>
  );
}
