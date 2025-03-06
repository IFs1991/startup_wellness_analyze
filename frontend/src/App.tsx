import { Routes, Route, Navigate } from 'react-router-dom';
import { ThemeProvider as MuiThemeProvider } from '@mui/material/styles';
import CssBaseline from '@mui/material/CssBaseline';
// ページ
import HomePage from './pages/HomePage';
import LoginPage from './pages/LoginPage';
import CompaniesPage from './pages/CompaniesPage';
import CompanyDetailPage from './pages/CompanyDetailPage';
import AnalysisPage from './pages/AnalysisPage';
import ReportsPage from './pages/ReportsPage';
import SettingsPage from './pages/SettingsPage';
import SubscriptionPage from './pages/SubscriptionPage';
import PricingPage from './pages/PricingPage';
// コンポーネント
import Layout from './components/layout/Layout';
import ProtectedRoute from './components/auth/ProtectedRoute';
// コンテキスト
import { AuthProvider } from './contexts/AuthContext';
import { SnackbarProvider } from './contexts/SnackbarContext';
// テーマ
import theme from './theme';
import { ThemeProvider } from './components/theme/ThemeProvider';

function App() {
  return (
    <MuiThemeProvider theme={theme}>
      <ThemeProvider>
        <CssBaseline />
        <SnackbarProvider>
          <AuthProvider>
            <Routes>
              {/* 認証不要のページ */}
              <Route path="/login" element={<LoginPage />} />
              <Route path="/pricing" element={<PricingPage />} />
              {/* 認証必須のページ */}
              <Route element={<Layout />}>
                <Route path="/" element={<Navigate to="/dashboard" replace />} />
                <Route
                  path="/dashboard"
                  element={
                    <ProtectedRoute>
                      <HomePage />
                    </ProtectedRoute>
                  }
                />
                <Route
                  path="/companies"
                  element={
                    <ProtectedRoute>
                      <CompaniesPage />
                    </ProtectedRoute>
                  }
                />
                <Route
                  path="/companies/:id"
                  element={
                    <ProtectedRoute>
                      <CompanyDetailPage />
                    </ProtectedRoute>
                  }
                />
                <Route
                  path="/analysis"
                  element={
                    <ProtectedRoute>
                      <AnalysisPage />
                    </ProtectedRoute>
                  }
                />
                <Route
                  path="/reports"
                  element={
                    <ProtectedRoute>
                      <ReportsPage />
                    </ProtectedRoute>
                  }
                />
                <Route
                  path="/settings"
                  element={
                    <ProtectedRoute>
                      <SettingsPage />
                    </ProtectedRoute>
                  }
                />
                <Route
                  path="/subscription"
                  element={
                    <ProtectedRoute>
                      <SubscriptionPage />
                    </ProtectedRoute>
                  }
                />
              </Route>
              {/* 存在しないパスへのリダイレクト */}
              <Route path="*" element={<Navigate to="/dashboard" replace />} />
            </Routes>
          </AuthProvider>
        </SnackbarProvider>
      </ThemeProvider>
    </MuiThemeProvider>
  );
}

export default App;