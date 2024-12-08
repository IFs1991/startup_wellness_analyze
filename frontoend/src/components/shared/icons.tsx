"use client"

import {
  Loader2,
  LogIn,
  LogOut,
  UserPlus,
  User,
  Settings,
  type LucideIcon,
} from "lucide-react"

export type Icon = LucideIcon

export const Icons = {
  spinner: Loader2,
  login: LogIn,
  logout: LogOut,
  register: UserPlus,
  user: User,
  settings: Settings,
}