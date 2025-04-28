"use client"

import { Button } from "@/components/ui/button"
import { ScrollArea } from "@/components/ui/scroll-area"
import {
  LayoutDashboard,
  BarChart,
  Building,
  FileText,
  Settings,
  ChevronLeft,
  ChevronRight,
  TrendingUp,
  Network,
  Activity,
  MessageSquare,
  Percent,
  ScatterChart,
  GitCompare,
  PieChart,
  BrainCircuit,
  LineChart,
  DollarSign,
  BarChart4,
  Users,
  Calculator,
  Workflow,
  Share2,
  Medal,
  Heart,
} from "lucide-react"
import Link from "next/link"
import { cn } from "@/lib/utils"
import { memo } from "react"

interface LeftSidebarProps {
  collapsed: boolean
  onToggleCollapse: () => void
}

// ナビゲーション項目を定数として定義
const navItems = [
  {
    id: "dashboard",
    label: "ダッシュボード",
    icon: <LayoutDashboard className="h-5 w-5" />,
    href: "/analysis/dashboard",
  },
  {
    id: "analysis",
    label: "詳細分析",
    icon: <BarChart className="h-5 w-5" />,
    children: [
      {
        id: "time-series",
        label: "時系列分析",
        icon: <TrendingUp className="h-4 w-4" />,
        href: "/analysis/time-series",
      },
      {
        id: "clustering",
        label: "クラスタリング",
        icon: <ScatterChart className="h-4 w-4" />,
        href: "/analysis/clustering",
      },
      {
        id: "correlation",
        label: "相関分析",
        icon: <Network className="h-4 w-4" />,
        href: "/analysis/correlation",
      },
      {
        id: "survival",
        label: "生存分析",
        icon: <Activity className="h-4 w-4" />,
        href: "/analysis/survival",
      },
      {
        id: "text-mining",
        label: "テキストマイニング",
        icon: <MessageSquare className="h-4 w-4" />,
        href: "/analysis/text-mining",
      },
      {
        id: "bayesian",
        label: "ベイジアン分析",
        icon: <Percent className="h-4 w-4" />,
        href: "/analysis/bayesian",
      },
      {
        id: "association",
        label: "アソシエーション分析",
        icon: <GitCompare className="h-4 w-4" />,
        href: "/analysis/association",
      },
      {
        id: "descriptive-stats",
        label: "記述統計",
        icon: <PieChart className="h-4 w-4" />,
        href: "/analysis/descriptive-stats",
      },
      {
        id: "causal-inference",
        label: "因果推論",
        icon: <BrainCircuit className="h-4 w-4" />,
        href: "/analysis/causal-inference",
      },
      {
        id: "pca",
        label: "主成分分析",
        icon: <LineChart className="h-4 w-4" />,
        href: "/analysis/pca",
      },
      {
        id: "financial",
        label: "財務分析",
        icon: <DollarSign className="h-4 w-4" />,
        href: "/analysis/financial",
      },
      {
        id: "market",
        label: "市場・競合分析",
        icon: <BarChart4 className="h-4 w-4" />,
        href: "/analysis/market",
      },
      {
        id: "team",
        label: "チーム・組織分析",
        icon: <Users className="h-4 w-4" />,
        href: "/analysis/team",
      },
      {
        id: "montecarlo",
        label: "モンテカルロシミュレーション",
        icon: <Calculator className="h-4 w-4" />,
        href: "/analysis/montecarlo",
      },
      {
        id: "sensitivity",
        label: "感度分析",
        icon: <Workflow className="h-4 w-4" />,
        href: "/analysis/sensitivity",
      },
      {
        id: "portfolio",
        label: "ポートフォリオネットワーク",
        icon: <Share2 className="h-4 w-4" />,
        href: "/analysis/portfolio",
      },
      {
        id: "vc-roi",
        label: "VC向けROI計算",
        icon: <Medal className="h-4 w-4" />,
        href: "/analysis/vc-roi",
      },
      {
        id: "health-index",
        label: "健康投資効果指数",
        icon: <Heart className="h-4 w-4" />,
        href: "/analysis/health-index",
      },
    ],
  },
  {
    id: "reports",
    label: "レポート",
    icon: <FileText className="h-5 w-5" />,
    href: "/reports",
  },
  {
    id: "companies",
    label: "企業管理",
    icon: <Building className="h-5 w-5" />,
    href: "/companies",
  },
]

const bottomNavItems = [
  {
    id: "settings",
    label: "設定",
    icon: <Settings className="h-5 w-5" />,
    href: "/settings",
  },
]

export const LeftSidebar = memo(function LeftSidebar({ collapsed, onToggleCollapse }: LeftSidebarProps) {
  return (
    <div
      className={cn(
        "relative flex h-full flex-col border-r border-background-lighter bg-background-light transition-all duration-300",
        collapsed ? "w-16" : "w-64",
      )}
    >
      <Button
        variant="ghost"
        size="icon"
        className="absolute -right-3 top-4 z-10 h-6 w-6 rounded-full border border-background-lighter bg-background-light"
        onClick={onToggleCollapse}
      >
        {collapsed ? <ChevronRight className="h-3 w-3" /> : <ChevronLeft className="h-3 w-3" />}
      </Button>

      <ScrollArea className="flex-1 px-3 py-4">
        <nav className="space-y-6">
          {navItems.map((item) => (
            <div key={item.id} className="space-y-2">
              {item.href ? (
                <Link
                  href={item.href}
                  className="flex items-center gap-3 rounded-md px-3 py-2 text-text-secondary hover:bg-background-lighter hover:text-text-primary"
                >
                  {item.icon}
                  {!collapsed && <span>{item.label}</span>}
                </Link>
              ) : (
                <>
                  <div className="flex items-center gap-3 px-3 py-2 text-text-secondary">
                    {item.icon}
                    {!collapsed && <span className="font-medium">{item.label}</span>}
                  </div>
                  {!collapsed && item.children && (
                    <div className="ml-4 space-y-1">
                      {item.children.map((child) => (
                        <Link
                          key={child.id}
                          href={child.href}
                          className="flex items-center gap-3 rounded-md px-3 py-2 text-sm text-text-secondary hover:bg-background-lighter hover:text-text-primary"
                        >
                          {child.icon}
                          <span>{child.label}</span>
                        </Link>
                      ))}
                    </div>
                  )}
                </>
              )}
            </div>
          ))}
        </nav>
      </ScrollArea>

      <div className="border-t border-background-lighter p-3">
        <nav>
          {bottomNavItems.map((item) => (
            <Link
              key={item.id}
              href={item.href}
              className="flex items-center gap-3 rounded-md px-3 py-2 text-text-secondary hover:bg-background-lighter hover:text-text-primary"
            >
              {item.icon}
              {!collapsed && <span>{item.label}</span>}
            </Link>
          ))}
        </nav>
      </div>
    </div>
  )
})

