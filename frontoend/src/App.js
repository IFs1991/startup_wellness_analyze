import React, { useState, useEffect, useContext } from 'react';
import { BrowserRouter as Router, Route, Routes, Navigate } from 'react-router-dom';
import { ThemeProvider, createTheme } from '@mui/material/styles';
import CssBaseline from '@mui/material/CssBaseline';
import AuthService from './services/AuthService';
import AuthContext from './context/AuthContext';

import Home from './pages/Home';
import Analysis from './pages/Analysis';
import Visualization from './pages/Visualization';
import Report from './pages/Report';
import LoginForm from './components/LoginForm';

const theme = createTheme({
  palette: {
    primary: {
      main: '#4285F4', // Google Blue
    },
    secondary: {
      main: '#EA4335', // Google Red
    },
  },
});

function App() {
  const [user, setUser] = useState(null);
  const { isAuthenticated, setIsAuthenticated } = useContext(AuthContext);

  useEffect(() => {
    const checkUserSession = async () => {
      try {
        const currentUser = await AuthService.getCurrentUser();
        setUser(currentUser);
        setIsAuthenticated(true);
      } catch (error) {
        console.error('Failed to check user session:', error);
      }
    };
    checkUserSession();
  }, []);

  const handleLoginSuccess = async (userData) => {
    setUser(userData);
    setIsAuthenticated(true);
  };

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Router>
        <Routes>
          <Route
            path="/login"
            element={!isAuthenticated ? <LoginForm onLoginSuccess={handleLoginSuccess} /> : <Navigate to="/" />}
          />
          <Route path="/" element={isAuthenticated ? <Home /> : <Navigate to="/login" />} />
          <Route path="/analysis" element={isAuthenticated ? <Analysis /> : <Navigate to="/login" />} />
          <Route path="/visualization" element={isAuthenticated ? <Visualization /> : <Navigate to="/login" />} />
          <Route path="/report" element={isAuthenticated ? <Report /> : <Navigate to="/login" />} />
        </Routes>
      </Router>
    </ThemeProvider>
  );
}

export default App;