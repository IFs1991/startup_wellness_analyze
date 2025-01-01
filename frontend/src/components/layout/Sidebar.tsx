import { cn } from '@/lib/utils';
import { routes } from '@/lib/routes';
import { NavLink } from 'react-router-dom';

export function Sidebar() {
  return (
    <div className="w-64 bg-card border-r border-border h-screen fixed left-0 top-0">
      <div className="p-4 border-b border-border">
        <h1 className="text-primary text-2xl font-semibold">ダッシュボード</h1>
      </div>
      <nav className="p-4">
        <ul className="space-y-2">
          {routes.map((route) => (
            <li key={route.path}>
              <NavLink
                to={route.path}
                className={({ isActive }) =>
                  cn(
                    "flex items-center space-x-3 px-4 py-2 rounded-lg",
                    "text-muted-foreground hover:bg-background hover:text-primary",
                    "transition-colors duration-200",
                    isActive && "bg-background text-primary"
                  )
                }
              >
                <route.icon className="h-5 w-5" />
                <span>{route.label}</span>
              </NavLink>
            </li>
          ))}
        </ul>
      </nav>
    </div>
  );
}