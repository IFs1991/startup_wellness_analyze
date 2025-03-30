import { createBrowserRouter } from 'react-router-dom'
import App from './App'
import DashboardPage from './pages/DashboardPage'
import CompaniesPage from './pages/CompaniesPage'
import CompanyDetailPage from './pages/CompanyDetailPage'
import LoginPage from './pages/LoginPage'
import NotFoundPage from './pages/NotFoundPage'
import HomePage from './pages/HomePage'
import ReportsPage from './pages/ReportsPage'
import AddStartupPage from './pages/AddStartupPage'
import AnalysisPage from './pages/AnalysisPage'
import PricingPage from './pages/PricingPage'
import { SettingsPage } from './pages/SettingsPage'
import { SubscriptionPage } from './pages/SubscriptionPage'

export const router = createBrowserRouter([
  {
    path: '/',
    element: <App />,
    errorElement: <NotFoundPage />,
    children: [
      {
        index: true,
        element: <HomePage />
      },
      {
        path: '/dashboard',
        element: <DashboardPage />
      },
      {
        path: '/companies',
        element: <CompaniesPage />
      },
      {
        path: '/companies/:id',
        element: <CompanyDetailPage />
      },
      {
        path: '/reports',
        element: <ReportsPage />
      },
      {
        path: '/startups/add',
        element: <AddStartupPage />
      },
      {
        path: '/analysis',
        element: <AnalysisPage />
      },
      {
        path: '/pricing',
        element: <PricingPage />
      },
      {
        path: '/settings',
        element: <SettingsPage />
      },
      {
        path: '/subscription',
        element: <SubscriptionPage />
      }
    ]
  },
  {
    path: '/login',
    element: <LoginPage />
  }
])