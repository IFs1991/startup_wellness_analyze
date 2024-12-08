import * as React from "react"
import Link from "next/link"
import { redirect } from "next/navigation"
import { createUserWithEmailAndPassword, updateProfile } from "firebase/auth"

import { auth } from "@/lib/firebase"
import { Icons } from "@/components/shared/icons"
import { Button } from "@/components/ui/button"
import {
  Card,
  CardContent,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle,
} from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { useToast } from "@/components/ui/use-toast"

export default function RegisterPage() {
  const [isLoading, setIsLoading] = React.useState<boolean>(false)
  const { toast } = useToast()

  async function onSubmit(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault()
    setIsLoading(true)

    const formData = new FormData(event.currentTarget)
    const name = formData.get("name") as string
    const email = formData.get("email") as string
    const password = formData.get("password") as string
    const confirmPassword = formData.get("confirmPassword") as string

    if (password !== confirmPassword) {
      toast({
        title: "エラー",
        description: "パスワードが一致しません",
        variant: "destructive",
      })
      setIsLoading(false)
      return
    }

    try {
      const userCredential = await createUserWithEmailAndPassword(
        auth,
        email,
        password
      )

      if (userCredential.user) {
        await updateProfile(userCredential.user, {
          displayName: name,
        })

        toast({
          title: "アカウント作成成功",
          description: "ダッシュボードにリダイレクトします",
        })
        redirect("/dashboard")
      }
    } catch (error) {
      toast({
        title: "エラー",
        description: "アカウントの作成に失敗しました。別のメールアドレスを試してください。",
        variant: "destructive",
      })
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <div className="flex h-screen w-screen items-center justify-center">
      <Card className="w-full max-w-md">
        <CardHeader className="space-y-1">
          <CardTitle className="text-2xl font-bold">アカウント作成</CardTitle>
          <CardDescription>
            必要な情報を入力して、アカウントを作成してください
          </CardDescription>
        </CardHeader>
        <form onSubmit={onSubmit}>
          <CardContent className="space-y-4">
            <div className="space-y-2">
              <Label htmlFor="name">名前</Label>
              <Input
                id="name"
                name="name"
                type="text"
                placeholder="山田 太郎"
                required
                disabled={isLoading}
              />
            </div>
            <div className="space-y-2">
              <Label htmlFor="email">メールアドレス</Label>
              <Input
                id="email"
                name="email"
                type="email"
                placeholder="name@example.com"
                required
                disabled={isLoading}
              />
            </div>
            <div className="space-y-2">
              <Label htmlFor="password">パスワード</Label>
              <Input
                id="password"
                name="password"
                type="password"
                required
                disabled={isLoading}
              />
            </div>
            <div className="space-y-2">
              <Label htmlFor="confirmPassword">パスワード（確認）</Label>
              <Input
                id="confirmPassword"
                name="confirmPassword"
                type="password"
                required
                disabled={isLoading}
              />
            </div>
          </CardContent>
          <CardFooter className="flex flex-col space-y-4">
            <Button
              type="submit"
              className="w-full"
              disabled={isLoading}
            >
              {isLoading && (
                <Icons.spinner className="mr-2 h-4 w-4 animate-spin" />
              )}
              アカウント作成
            </Button>
            <div className="text-sm text-center text-muted-foreground">
              すでにアカウントをお持ちの場合は{" "}
              <Link
                href="/login"
                className="text-primary underline-offset-4 hover:underline"
              >
                ログイン
              </Link>
              {" "}してください
            </div>
          </CardFooter>
        </form>
      </Card>
    </div>
  )
}