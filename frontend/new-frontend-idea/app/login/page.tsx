"use client";

import { useState } from 'react';
import Link from 'next/link';
import { useRouter } from 'next/navigation';
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { useAuth } from '@/hooks/useAuth';
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Loader2 } from 'lucide-react';

export default function LoginPage() {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const { login, user } = useAuth();
  const router = useRouter();

  // 既にログインしている場合はダッシュボードへリダイレクト
  if (user) {
    router.push('/');
    return null; // リダイレクト中は何も表示しない
  }

  const handleLogin = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);
    setIsLoading(true);

    try {
      await login(email, password);
      // ログイン成功時の処理は useAuth フック内でリダイレクトなどを担当すると想定
      // もしくはここで router.push('/') を呼ぶ
      router.push('/'); // 例: ログイン成功後にダッシュボードへ
    } catch (err: any) {
      console.error("Login failed:", err);
      // Firebaseのエラーコードに基づいてメッセージを設定
      let errorMessage = "ログインに失敗しました。メールアドレスまたはパスワードを確認してください。";
      if (err.code === 'auth/user-not-found' || err.code === 'auth/wrong-password' || err.code === 'auth/invalid-credential') {
        errorMessage = "メールアドレスまたはパスワードが間違っています。";
      } else if (err.code === 'auth/invalid-email') {
        errorMessage = "メールアドレスの形式が正しくありません。";
      } else if (err.code === 'auth/too-many-requests') {
        errorMessage = "試行回数が多すぎます。後でもう一度お試しください。";
      }
      setError(errorMessage);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="flex items-center justify-center min-h-screen bg-background">
      <Card className="mx-auto max-w-sm w-full">
        <CardHeader>
          <CardTitle className="text-2xl text-center">ログイン</CardTitle>
          <CardDescription className="text-center">
            アカウント情報を入力してください
          </CardDescription>
        </CardHeader>
        <CardContent>
          <form onSubmit={handleLogin} className="space-y-4">
            {error && (
              <Alert variant="destructive">
                <AlertTitle>エラー</AlertTitle>
                <AlertDescription>{error}</AlertDescription>
              </Alert>
            )}
            <div className="space-y-2">
              <Label htmlFor="email">メールアドレス</Label>
              <Input
                id="email"
                type="email"
                placeholder="mail@example.com"
                required
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                disabled={isLoading}
              />
            </div>
            <div className="space-y-2">
              <div className="flex items-center">
                <Label htmlFor="password">パスワード</Label>
                <Link
                  href="/password-reset" // パスワードリセットページへのパス
                  className="ml-auto inline-block text-sm underline"
                >
                  パスワードをお忘れですか？
                </Link>
              </div>
              <Input
                id="password"
                type="password"
                required
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                disabled={isLoading}
              />
            </div>
            <Button type="submit" className="w-full" disabled={isLoading}>
              {isLoading ? (
                <Loader2 className="mr-2 h-4 w-4 animate-spin" />
              ) : (
                "ログイン"
              )}
            </Button>
            {/* TODO: ソーシャルログインボタンを追加する場合はここに */}
          </form>
          <div className="mt-4 text-center text-sm">
            アカウントをお持ちでないですか？{" "}
            <Link href="/register" className="underline"> {/* 登録ページへのパス */}
              アカウント登録
            </Link>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}