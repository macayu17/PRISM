import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { ThemeProvider } from './context/ThemeContext';
import Layout from './components/Layout';
import HomePage from './pages/HomePage';
import AssessmentPage from './pages/AssessmentPage';
import AboutPage from './pages/AboutPage';
import DocumentsPage from './pages/DocumentsPage';
import TwinPage from './pages/TwinPage';

export default function App() {
  return (
    <ThemeProvider>
      <BrowserRouter>
        <Routes>
          <Route element={<Layout />}>
            <Route path="/" element={<HomePage />} />
            <Route path="/assessment" element={<AssessmentPage />} />
            <Route path="/about" element={<AboutPage />} />
            <Route path="/documents" element={<DocumentsPage />} />
            <Route path="/twin" element={<TwinPage />} />
          </Route>
        </Routes>
      </BrowserRouter>
    </ThemeProvider>
  );
}
