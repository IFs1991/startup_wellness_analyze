import Link from "next/link"
import { cn } from "@/lib/utils"

interface NavItem {
  title: string
  href: string
  description: string
}

const items: NavItem[] = [
  {
    title: "ダッシュボード",
    href: "/dashboard",
    description: "アクティビティの概要を確認",
  },
  {
    title: "グラフ",
    href: "/graphs",
    description: "データの可視化と分析",
  },
]

export function MainNav() {
  return (
    <nav className="flex items-center space-x-6">
      {items.map((item) => (
        <Link
          key={item.href}
          href={item.href}
          className={cn(
            "text-sm font-medium transition-colors hover:text-primary",
          )}
        >
          {item.title}
        </Link>
      ))}
    </nav>
  )
}