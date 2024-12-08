"use client"

import * as React from "react"
import { Moon, Sun } from "lucide-react"
import { useTheme } from "next-themes"

import { Button } from "@/components/ui/button"
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu"

export function ModeToggle() {
  // next-themesのuseThemeフックからsetTheme関数を取得
  // この関数を使用してテーマを動的に変更します
  const { setTheme } = useTheme()

  return (
    <DropdownMenu>
      {/* ドロップダウンを開くためのトリガーボタン */}
      <DropdownMenuTrigger asChild>
        {/* アイコンを表示するボタン */}
        <Button variant="outline" size="icon">
          {/* ライトモードで表示される太陽のアイコン */}
          <Sun
            className="h-[1.2rem] w-[1.2rem] rotate-0 scale-100 transition-all dark:-rotate-90 dark:scale-0"
          />
          {/* ダークモードで表示される月のアイコン */}
          <Moon
            className="absolute h-[1.2rem] w-[1.2rem] rotate-90 scale-0 transition-all dark:rotate-0 dark:scale-100"
          />
          {/* アクセシビリティのための非表示テキスト */}
          <span className="sr-only">テーマを切り替える</span>
        </Button>
      </DropdownMenuTrigger>

      {/* テーマ選択のドロップダウンメニュー */}
      <DropdownMenuContent align="end">
        {/* ライトモードを選択するメニュー項目 */}
        <DropdownMenuItem onClick={() => setTheme("light")}>
          ライト
        </DropdownMenuItem>
        {/* ダークモードを選択するメニュー項目 */}
        <DropdownMenuItem onClick={() => setTheme("dark")}>
          ダーク
        </DropdownMenuItem>
        {/* システムテーマを選択するメニュー項目 */}
        <DropdownMenuItem onClick={() => setTheme("system")}>
          システム
        </DropdownMenuItem>
      </DropdownMenuContent>
    </DropdownMenu>
  )
}