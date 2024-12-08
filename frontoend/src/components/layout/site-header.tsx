import { MainNav as MainNavigation } from "./main-nav"
import { UserNav as UserNavigation } from "./user-nav"
import { ModeToggle } from "@/components/ui/mode-toggle"
import { Button } from "@/components/ui/button"
import { useAuth } from "@/lib/auth"
import {
 DropdownMenu,
 DropdownMenuContent,
 DropdownMenuItem,
 DropdownMenuLabel,
 DropdownMenuSeparator,
 DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu"
import Link from "next/link"

// 型定義の追加
interface User {
 name: string;
 email: string;
}

interface Auth {
 user: User | null;
 signOut: () => Promise<void>;
}

// コンポーネント名の競合を解決
const MainNav = MainNavigation
const UserNav = UserNavigation

interface HeaderProps {
 className?: string
}

export function SiteHeader({ className }: HeaderProps) {
 const { user, signOut } = useAuth() as Auth

 return (
   <header className={`w-full border-b bg-background ${className}`}>
     <div className="container flex h-16 items-center space-x-4 sm:justify-between sm:space-x-0">
       {/* メインナビゲーション領域 */}
       <MainNav />
       {/* 右側のナビゲーション領域 */}
       <div className="flex flex-1 items-center justify-end space-x-4">
         {/* ダークモードトグル */}
         <nav className="flex items-center space-x-2">
           <ModeToggle />
           {/* ユーザーがログインしている場合のメニュー */}
           {user ? (
             <DropdownMenu>
               <DropdownMenuTrigger asChild>
                 <Button variant="ghost" className="flex items-center space-x-2">
                   <UserNav />
                 </Button>
               </DropdownMenuTrigger>
               <DropdownMenuContent align="end">
                 <DropdownMenuLabel>
                   <div className="flex flex-col space-y-1">
                     <p className="text-sm font-medium leading-none">{user.name}</p>
                     <p className="text-xs leading-none text-muted-foreground">
                       {user.email}
                     </p>
                   </div>
                 </DropdownMenuLabel>
                 <DropdownMenuSeparator />
                 <DropdownMenuItem>
                   <Link href="/dashboard">Dashboard</Link>
                 </DropdownMenuItem>
                 <DropdownMenuItem onClick={() => signOut()}>
                   Sign out
                 </DropdownMenuItem>
               </DropdownMenuContent>
             </DropdownMenu>
           ) : (
             // 未ログインの場合のリンク
             <Link href="/login" className="text-sm font-medium">
               Sign in
             </Link>
           )}
         </nav>
       </div>
     </div>
   </header>
 )
}