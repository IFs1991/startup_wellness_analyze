import type React from "react"
export function Header({ children }: { children: React.ReactNode }) {
  return (
    <header className="flex h-16 items-center justify-between border-b border-background-lighter bg-background-light px-4">
      {children}
    </header>
  )
}

