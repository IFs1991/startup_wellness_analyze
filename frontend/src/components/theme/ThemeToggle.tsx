import { Moon, Sun } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { useTheme } from './ThemeProvider';

export function ThemeToggle() {
  const { theme, toggleTheme } = useTheme();

  return (
    <Button
      variant="ghost"
      size="icon"
      onClick={toggleTheme}
      className="w-10 h-10 rounded-full"
    >
      {theme === 'light' ? (
        <Moon className="h-5 w-5 text-[#757575]" />
      ) : (
        <Sun className="h-5 w-5 text-[#FFC107]" />
      )}
    </Button>
  );
}