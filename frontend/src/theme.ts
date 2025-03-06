import { createTheme } from '@mui/material/styles';
import { jaJP } from '@mui/material/locale';

// テーマカラーの定義
const themeColors = {
  primary: {
    main: '#3b82f6', // ブルー
    light: '#60a5fa',
    dark: '#2563eb',
    contrastText: '#ffffff',
  },
  secondary: {
    main: '#10b981', // エメラルドグリーン
    light: '#34d399',
    dark: '#059669',
    contrastText: '#ffffff',
  },
  error: {
    main: '#ef4444',
    light: '#f87171',
    dark: '#b91c1c',
    contrastText: '#ffffff',
  },
  warning: {
    main: '#f59e0b',
    light: '#fbbf24',
    dark: '#d97706',
    contrastText: '#ffffff',
  },
  info: {
    main: '#3b82f6',
    light: '#60a5fa',
    dark: '#2563eb',
    contrastText: '#ffffff',
  },
  success: {
    main: '#10b981',
    light: '#34d399',
    dark: '#059669',
    contrastText: '#ffffff',
  },
  grey: {
    50: '#f9fafb',
    100: '#f3f4f6',
    200: '#e5e7eb',
    300: '#d1d5db',
    400: '#9ca3af',
    500: '#6b7280',
    600: '#4b5563',
    700: '#374151',
    800: '#1f2937',
    900: '#111827',
  },
  text: {
    primary: '#f3f4f6',
    secondary: '#d1d5db',
    disabled: '#6b7280',
  },
  background: {
    default: '#111827',
    paper: '#1f2937',
  },
  divider: '#374151',
};

// ダークモードで使用する色
const darkThemeColors = {
  ...themeColors,
  text: {
    primary: '#f3f4f6',
    secondary: '#d1d5db',
    disabled: '#6b7280',
  },
  background: {
    default: '#111827',
    paper: '#1f2937',
  },
  divider: '#374151',
};

// テーマの作成
const theme = createTheme(
  {
    palette: {
      ...darkThemeColors,
      mode: 'dark',
    },
    typography: {
      fontFamily: [
        'Hiragino Sans',
        'Hiragino Kaku Gothic ProN',
        'Noto Sans JP',
        'Meiryo',
        'sans-serif',
        'Apple Color Emoji',
        'Segoe UI Emoji',
      ].join(','),
      h1: {
        fontSize: '2.5rem',
        fontWeight: 700,
      },
      h2: {
        fontSize: '2rem',
        fontWeight: 700,
      },
      h3: {
        fontSize: '1.75rem',
        fontWeight: 600,
      },
      h4: {
        fontSize: '1.5rem',
        fontWeight: 600,
      },
      h5: {
        fontSize: '1.25rem',
        fontWeight: 600,
      },
      h6: {
        fontSize: '1rem',
        fontWeight: 600,
      },
      subtitle1: {
        fontSize: '1rem',
        fontWeight: 500,
      },
      subtitle2: {
        fontSize: '0.875rem',
        fontWeight: 500,
      },
      body1: {
        fontSize: '1rem',
      },
      body2: {
        fontSize: '0.875rem',
      },
      button: {
        textTransform: 'none',
        fontWeight: 600,
      },
    },
    shape: {
      borderRadius: 8,
    },
    shadows: [
      'none',
      '0px 2px 1px -1px rgba(0,0,0,0.2), 0px 1px 1px 0px rgba(0,0,0,0.14), 0px 1px 3px 0px rgba(0,0,0,0.12)',
      '0px 3px 1px -2px rgba(0,0,0,0.2), 0px 2px 2px 0px rgba(0,0,0,0.14), 0px 1px 5px 0px rgba(0,0,0,0.12)',
      '0px 3px 3px -2px rgba(0,0,0,0.2), 0px 3px 4px 0px rgba(0,0,0,0.14), 0px 1px 8px 0px rgba(0,0,0,0.12)',
      '0px 2px 4px -1px rgba(0,0,0,0.2), 0px 4px 5px 0px rgba(0,0,0,0.14), 0px 1px 10px 0px rgba(0,0,0,0.12)',
      '0px 3px 5px -1px rgba(0,0,0,0.2), 0px 5px 8px 0px rgba(0,0,0,0.14), 0px 1px 14px 0px rgba(0,0,0,0.12)',
      '0px 3px 5px -1px rgba(0,0,0,0.2), 0px 6px 10px 0px rgba(0,0,0,0.14), 0px 1px 18px 0px rgba(0,0,0,0.12)',
      '0px 4px 5px -2px rgba(0,0,0,0.2), 0px 7px 10px 1px rgba(0,0,0,0.14), 0px 2px 16px 1px rgba(0,0,0,0.12)',
      '0px 5px 5px -3px rgba(0,0,0,0.2), 0px 8px 10px 1px rgba(0,0,0,0.14), 0px 3px 14px 2px rgba(0,0,0,0.12)',
      '0px 5px 6px -3px rgba(0,0,0,0.2), 0px 9px 12px 1px rgba(0,0,0,0.14), 0px 3px 16px 2px rgba(0,0,0,0.12)',
      '0px 6px 6px -3px rgba(0,0,0,0.2), 0px 10px 14px 1px rgba(0,0,0,0.14), 0px 4px 18px 3px rgba(0,0,0,0.12)',
      '0px 6px 7px -4px rgba(0,0,0,0.2), 0px 11px 15px 1px rgba(0,0,0,0.14), 0px 4px 20px 3px rgba(0,0,0,0.12)',
      '0px 7px 8px -4px rgba(0,0,0,0.2), 0px 12px 17px 2px rgba(0,0,0,0.14), 0px 5px 22px 4px rgba(0,0,0,0.12)',
      '0px 7px 8px -4px rgba(0,0,0,0.2), 0px 13px 19px 2px rgba(0,0,0,0.14), 0px 5px 24px 4px rgba(0,0,0,0.12)',
      '0px 7px 9px -4px rgba(0,0,0,0.2), 0px 14px 21px 2px rgba(0,0,0,0.14), 0px 5px 26px 4px rgba(0,0,0,0.12)',
      '0px 8px 9px -5px rgba(0,0,0,0.2), 0px 15px 22px 2px rgba(0,0,0,0.14), 0px 6px 28px 5px rgba(0,0,0,0.12)',
      '0px 8px 10px -5px rgba(0,0,0,0.2), 0px 16px 24px 2px rgba(0,0,0,0.14), 0px 6px 30px 5px rgba(0,0,0,0.12)',
      '0px 8px 11px -5px rgba(0,0,0,0.2), 0px 17px 26px 2px rgba(0,0,0,0.14), 0px 6px 32px 5px rgba(0,0,0,0.12)',
      '0px 9px 11px -5px rgba(0,0,0,0.2), 0px 18px 28px 2px rgba(0,0,0,0.14), 0px 7px 34px 6px rgba(0,0,0,0.12)',
      '0px 9px 12px -6px rgba(0,0,0,0.2), 0px 19px 29px 2px rgba(0,0,0,0.14), 0px 7px 36px 6px rgba(0,0,0,0.12)',
      '0px 10px 13px -6px rgba(0,0,0,0.2), 0px 20px 31px 3px rgba(0,0,0,0.14), 0px 8px 38px 7px rgba(0,0,0,0.12)',
      '0px 10px 13px -6px rgba(0,0,0,0.2), 0px 21px 33px 3px rgba(0,0,0,0.14), 0px 8px 40px 7px rgba(0,0,0,0.12)',
      '0px 10px 14px -6px rgba(0,0,0,0.2), 0px 22px 35px 3px rgba(0,0,0,0.14), 0px 8px 42px 7px rgba(0,0,0,0.12)',
      '0px 11px 14px -7px rgba(0,0,0,0.2), 0px 23px 36px 3px rgba(0,0,0,0.14), 0px 9px 44px 8px rgba(0,0,0,0.12)',
      '0px 11px 15px -7px rgba(0,0,0,0.2), 0px 24px 38px 3px rgba(0,0,0,0.14), 0px 9px 46px 8px rgba(0,0,0,0.12)',
    ],
    components: {
      MuiAppBar: {
        styleOverrides: {
          root: {
            backgroundColor: '#111827',
            boxShadow: 'none',
            borderBottom: '1px solid #374151',
          },
        },
      },
      MuiDrawer: {
        styleOverrides: {
          paper: {
            backgroundColor: '#111827',
            borderRight: '1px solid #374151',
          },
        },
      },
      MuiButton: {
        styleOverrides: {
          root: {
            borderRadius: 8,
            padding: '8px 16px',
            fontWeight: 600,
          },
          contained: {
            boxShadow: 'none',
            '&:hover': {
              boxShadow: '0px 2px 4px rgba(0, 0, 0, 0.3)',
            },
          },
        },
      },
      MuiCard: {
        styleOverrides: {
          root: {
            backgroundColor: '#1f2937',
            boxShadow: '0px 4px 10px rgba(0, 0, 0, 0.2)',
            borderRadius: 12,
            border: '1px solid #374151',
          },
        },
      },
      MuiCardHeader: {
        styleOverrides: {
          root: {
            padding: '16px 24px',
          },
        },
      },
      MuiCardContent: {
        styleOverrides: {
          root: {
            padding: '24px',
            '&:last-child': {
              paddingBottom: '24px',
            },
          },
        },
      },
      MuiTextField: {
        styleOverrides: {
          root: {
            '& .MuiOutlinedInput-root': {
              borderRadius: 8,
              backgroundColor: '#1f2937',
              '& fieldset': {
                borderColor: '#374151',
              },
              '&:hover fieldset': {
                borderColor: '#3b82f6',
              },
            },
          },
        },
      },
      MuiInputBase: {
        styleOverrides: {
          root: {
            backgroundColor: '#1f2937',
          },
        },
      },
      MuiTableHead: {
        styleOverrides: {
          root: {
            backgroundColor: '#1f2937',
          },
        },
      },
      MuiTableCell: {
        styleOverrides: {
          root: {
            padding: '16px',
            borderBottom: '1px solid #374151',
          },
          head: {
            fontWeight: 600,
            color: darkThemeColors.text.primary,
            backgroundColor: '#172033',
          },
        },
      },
      MuiChip: {
        styleOverrides: {
          root: {
            fontWeight: 500,
            backgroundColor: '#2563eb',
            color: '#fff',
          },
          filled: {
            backgroundColor: '#2563eb',
          },
        },
      },
      MuiTabs: {
        styleOverrides: {
          root: {
            borderBottom: '1px solid #374151',
          },
          indicator: {
            backgroundColor: '#3b82f6',
          },
        },
      },
      MuiTab: {
        styleOverrides: {
          root: {
            color: '#d1d5db',
            '&.Mui-selected': {
              color: '#3b82f6',
            },
          },
        },
      },
      MuiDialog: {
        styleOverrides: {
          paper: {
            backgroundColor: '#1f2937',
            borderRadius: 12,
          },
        },
      },
      MuiDialogTitle: {
        styleOverrides: {
          root: {
            padding: '16px 24px',
          },
        },
      },
      MuiMenu: {
        styleOverrides: {
          paper: {
            backgroundColor: '#1f2937',
            borderRadius: 8,
            border: '1px solid #374151',
          },
        },
      },
      MuiMenuItem: {
        styleOverrides: {
          root: {
            '&:hover': {
              backgroundColor: '#374151',
            },
          },
        },
      },
    },
  },
  jaJP
);

export default theme;