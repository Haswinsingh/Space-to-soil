import React from 'react'
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom'
import LandingPage from './pages/LandingPage'
import { LoginPage, RegisterPage, ForgotPasswordPage } from './pages/AuthPages'
import Dashboard from './pages/Dashboard'
import DashboardOverview from './pages/DashboardOverview'
import Upload from './pages/Upload'
import Predictions from './pages/Predictions'
import ModelBenchmarking from './pages/ModelBenchmarking'
import Reports from './pages/Reports'
import Settings from './pages/Settings'
import { AuthProvider } from './context/AuthContext'
import ProtectedRoute from './components/ProtectedRoute'

function App() {
  return (
    <AuthProvider>
      <BrowserRouter>
        <Routes>
          {/* Public Routes */}
          <Route path="/" element={<LandingPage />} />
          <Route path="/login" element={<LoginPage />} />
          <Route path="/register" element={<RegisterPage />} />
          <Route path="/forgot-password" element={<ForgotPasswordPage />} />

          {/* Protected Dashboard Layout with Nested Routes */}
          <Route path="/dashboard/*" element={
            <ProtectedRoute>
              <Dashboard />
            </ProtectedRoute>
          }>
            <Route index element={<DashboardOverview />} />
            <Route path="upload" element={<Upload />} />
            <Route path="predictions" element={<Predictions />} />
            <Route path="ml-benchmark" element={<ModelBenchmarking />} />
            <Route path="reports" element={<Reports />} />
            <Route path="settings" element={<Settings />} />
          </Route>

          {/* Catch-all */}
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </BrowserRouter>
    </AuthProvider>
  );
}

export default App;
