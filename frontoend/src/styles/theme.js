import { createTheme } from '@mui/material/styles';

const theme = createTheme({
  palette: {
    primary: {
      main: '#4285F4', // Google Blue
    },
    secondary: {
      main: '#EA4335', // Google Red
    },
  },
  typography: {
    fontFamily: 'Roboto, sans-serif',
  },
  components: {
    // 各コンポーネントのスタイルを上書き
    MuiButton: {
      styleOverrides: {
        root: {
          borderRadius: '4px',
        },
      },
    },
  },
});

export default theme;