import { Routes, Route, Navigate } from 'react-router-dom'
import Navbar from './components/Navbar'
import DatasetsPage from './pages/DatasetsPage'
import ExperimentsPage from './pages/ExperimentsPage'
import ExperimentDetailPage from './pages/ResultsPage'
// import ComparisonPage from './pages/ComparisonPage'
import LoginPage from './pages/LoginPage'
import RegisterPage from './pages/RegisterPage'
import MyDataPage from './pages/MyDataPage'

export default function App() {
  return (
    <>
      <Navbar />
      <Routes>
        <Route path="/" element={<DatasetsPage />} />
        <Route path="/experiments" element={<ExperimentsPage />} />
        <Route path="/experiments/:id" element={<ExperimentDetailPage />} />
        {/* <Route path="/results" element={<ComparisonPage />} /> */}
        <Route path="/my-data" element={<MyDataPage />} />
        <Route path="/login" element={<LoginPage />} />
        <Route path="/register" element={<RegisterPage />} />
        {/* <Route path="/comparison" element={<Navigate to="/results" replace />} /> */}
        <Route path="/results" element={<Navigate to="/" replace />} />
        <Route path="/comparison" element={<Navigate to="/" replace />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </>
  )
}
