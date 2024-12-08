import { cn } from "@/lib/utils"
import React from "react"

interface PageHeaderProps extends React.HTMLAttributes<HTMLDivElement> {
  as?: React.ElementType
}

interface PageHeaderDescriptionProps
  extends React.HTMLAttributes<HTMLParagraphElement> {}

interface PageHeaderHeadingProps
  extends React.HTMLAttributes<HTMLHeadingElement> {}

export function PageHeader({
  className,
  children,
  as: Component = "section",
  ...props
}: PageHeaderProps) {
  return (
    <Component
      className={cn("grid gap-1", className)}
      {...props}
    >
      {children}
    </Component>
  )
}

export function PageHeaderHeading({
  className,
  children,
  ...props
}: PageHeaderHeadingProps) {
  return (
    <h1
      className={cn(
        "text-3xl font-bold leading-tight tracking-tighter md:text-4xl",
        className
      )}
      {...props}
    >
      {children}
    </h1>
  )
}

export function PageHeaderDescription({
  className,
  children,
  ...props
}: PageHeaderDescriptionProps) {
  return (
    <p
      className={cn(
        "text-muted-foreground text-lg",
        className
      )}
      {...props}
    >
      {children}
    </p>
  )
}