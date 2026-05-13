import { lazy, Suspense } from 'react';
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { ThemeProvider } from './context/ThemeContext';
import Layout from './components/Layout';

const HomePage = lazy(() => import('./pages/HomePage'));
const AssessmentPage = lazy(() => import('./pages/AssessmentPage'));
const AboutPage = lazy(() => import('./pages/AboutPage'));
const DocumentsPage = lazy(() => import('./pages/DocumentsPage'));
const TwinPage = lazy(() => import('./pages/TwinPage'));

function RouteFallback() {
  return (
    <div className="mx-auto flex min-h-[50vh] w-full max-w-4xl items-center justify-center px-6 text-sm font-semibold uppercase text-slate-400">
      Loading clinical workspace...
    </div>
  );
}

export default function App() {
  return (
    <ThemeProvider>
      <BrowserRouter>
        <Suspense fallback={<RouteFallback />}>
          <Routes>
            <Route element={<Layout />}>
              <Route path="/" element={<HomePage />} />
              <Route path="/assessment" element={<AssessmentPage />} />
              <Route path="/about" element={<AboutPage />} />
              <Route path="/documents" element={<DocumentsPage />} />
              <Route path="/twin" element={<TwinPage />} />
            </Route>
          </Routes>
        </Suspense>
      </BrowserRouter>
    </ThemeProvider>
  );
}
