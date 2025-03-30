// import { Link } from 'react-router-dom';
import { Bell, Search, /* Menu, */ Settings, LogOut, User } from 'lucide-react';
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuGroup,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu';
import { Input } from '@/components/ui/input';
import { Avatar, AvatarFallback, AvatarImage } from '@/components/ui/avatar';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';

// 認証コンテキストの代替モック
const useAuth = () => {
  return {
    user: {
      displayName: '山田 太郎',
      email: 'user@example.com'
    },
    signOut: () => console.log('サインアウト')
  };
};

const Header = () => {
  const { user, signOut } = useAuth();

  const handleSignOut = () => {
    signOut();
  };

  const userInitial = user?.displayName ? user.displayName.charAt(0) : 'U';

  return (
    <div className="flex w-full items-center justify-between">
      <div className="relative max-w-md w-full md:flex items-center hidden">
        <Search className="absolute left-2.5 h-4 w-4 text-muted-foreground" />
        <Input
          type="search"
          placeholder="検索..."
          className="w-full pl-8 rounded-md"
        />
      </div>

      <div className="ml-auto flex items-center gap-3">
        <Button variant="ghost" size="icon" className="relative">
          <Bell className="h-5 w-5" />
          <Badge className="absolute -top-1 -right-1 flex h-5 w-5 items-center justify-center rounded-full">3</Badge>
          <span className="sr-only">通知</span>
        </Button>

        <DropdownMenu>
          <DropdownMenuTrigger asChild>
            <Button variant="ghost" className="relative h-9 w-9 rounded-full">
              <Avatar className="h-9 w-9">
                <AvatarImage src="" alt={user?.displayName} />
                <AvatarFallback>{userInitial}</AvatarFallback>
              </Avatar>
            </Button>
          </DropdownMenuTrigger>
          <DropdownMenuContent className="w-56" align="end" forceMount>
            <DropdownMenuLabel className="font-normal">
              <div className="flex flex-col space-y-1">
                <p className="text-sm font-medium leading-none">{user?.displayName}</p>
                <p className="text-xs leading-none text-muted-foreground">{user?.email}</p>
              </div>
            </DropdownMenuLabel>
            <DropdownMenuSeparator />
            <DropdownMenuGroup>
              <DropdownMenuItem>
                <User className="mr-2 h-4 w-4" />
                <span>プロフィール</span>
              </DropdownMenuItem>
              <DropdownMenuItem>
                <Settings className="mr-2 h-4 w-4" />
                <span>設定</span>
              </DropdownMenuItem>
            </DropdownMenuGroup>
            <DropdownMenuSeparator />
            <DropdownMenuItem onClick={handleSignOut} className="text-destructive focus:text-destructive">
              <LogOut className="mr-2 h-4 w-4" />
              <span>ログアウト</span>
            </DropdownMenuItem>
          </DropdownMenuContent>
        </DropdownMenu>
      </div>
    </div>
  );
};

export default Header;