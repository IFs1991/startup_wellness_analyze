import React, { createContext, useContext, useState, ReactNode } from 'react';
import { Alert, Snackbar, AlertColor } from '@mui/material';

interface SnackbarMessage {
  message: string;
  severity: AlertColor;
  autoHideDuration?: number;
}

interface SnackbarContextType {
  showSnackbar: (message: string, severity?: AlertColor, autoHideDuration?: number) => void;
}

const SnackbarContext = createContext<SnackbarContextType | undefined>(undefined);

interface SnackbarProviderProps {
  children: ReactNode;
}

export const SnackbarProvider: React.FC<SnackbarProviderProps> = ({ children }) => {
  const [open, setOpen] = useState(false);
  const [snackbarMessage, setSnackbarMessage] = useState<SnackbarMessage>({
    message: '',
    severity: 'info',
    autoHideDuration: 6000,
  });

  const showSnackbar = (
    message: string,
    severity: AlertColor = 'info',
    autoHideDuration: number = 6000
  ) => {
    setSnackbarMessage({
      message,
      severity,
      autoHideDuration,
    });
    setOpen(true);
  };

  const handleClose = (event?: React.SyntheticEvent | Event, reason?: string) => {
    if (reason === 'clickaway') {
      return;
    }
    setOpen(false);
  };

  return (
    <SnackbarContext.Provider value={{ showSnackbar }}>
      {children}
      <Snackbar
        open={open}
        autoHideDuration={snackbarMessage.autoHideDuration}
        onClose={handleClose}
        anchorOrigin={{ vertical: 'bottom', horizontal: 'center' }}
      >
        <Alert onClose={handleClose} severity={snackbarMessage.severity} sx={{ width: '100%' }}>
          {snackbarMessage.message}
        </Alert>
      </Snackbar>
    </SnackbarContext.Provider>
  );
};

// カスタムフック
export const useSnackbar = (): SnackbarContextType => {
  const context = useContext(SnackbarContext);
  if (context === undefined) {
    throw new Error('useSnackbar must be used within a SnackbarProvider');
  }
  return context;
};