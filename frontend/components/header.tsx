import type React from "react"
export function Header({ children, className }: { children: React.ReactNode; className?: string }) {
  return (
    <header className={`flex h-16 items-center justify-between border-b border-background-lighter bg-background-light px-4${className ? ` ${className}` : ""}`}>
      {children}
    </header>
  )
}

