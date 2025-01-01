import { useTheme } from '@/components/theme/ThemeProvider';

export interface ChartTheme {
  axis: {
    stroke: string;
    fontSize: number;
  };
  grid: {
    stroke: string;
  };
  tooltip: {
    background: string;
    border: string;
    color: string;
  };
}

export function useChartTheme(): ChartTheme {
  const { theme } = useTheme();
  const isDark = theme === 'dark';

  return {
    axis: {
      stroke: isDark ? '#E0E0E0' : '#757575',
      fontSize: 12,
    },
    grid: {
      stroke: isDark ? '#333333' : '#EEEEEE',
    },
    tooltip: {
      background: isDark ? '#1F1F1F' : '#FFFFFF',
      border: isDark ? '#333333' : '#EEEEEE',
      color: isDark ? '#E0E0E0' : '#212121',
    },
  };
}