import * as React from "react"
import Link from "next/link"
import { redirect } from "next/navigation"
import { signInWithEmailAndPassword } from "firebase/auth"

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

export default function LoginPage() {
  const [isLoading, setIsLoading] = React.useState<boolean>(false)
  const { toast } = useToast()

  async function onSubmit(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault()
    setIsLoading(true)

    const formData = new FormData(event.currentTarget)
    const email = formData.get("email") as string
    const password = formData.get("password") as string

    try {
      await signInWithEmailAndPassword(auth, email, password)
      toast({
        title: "ログイン成功",
        description: "ダッシュボードにリダイレクトします",
      })
      redirect("/dashboard")
    } catch (error) {
      toast({
        title: "エラー",
        description: "メールアドレスまたはパスワードが正しくありません",
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
          <CardTitle className="text-2xl font-bold">ログイン</CardTitle>
          <CardDescription>
            アカウントにログインして、ダッシュボードにアクセスしてください
          </CardDescription>
        </CardHeader>
        <form onSubmit={onSubmit}>
          <CardContent className="space-y-4">
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
              ログイン
            </Button>
            <div className="text-sm text-center text-muted-foreground">
              アカウントをお持ちでない場合は{" "}
              <Link
                href="/register"
                className="text-primary underline-offset-4 hover:underline"
              >
                新規登録
              </Link>
              {" "}してください
            </div>
          </CardFooter>
        </form>
      </Card>
    </div>
  )
}