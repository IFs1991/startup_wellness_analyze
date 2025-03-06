import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { signInWithEmailAndPassword } from 'firebase/auth';
import { auth } from '../firebase/config';
import TrialSelectionModal from '../components/subscription/TrialSelectionModal';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Card } from '@/components/ui/card';
import { Separator } from '@/components/ui/separator';
import {
  Building,
  Lock,
  Star,
  CreditCard
} from 'lucide-react';

const LoginPage: React.FC = () => {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const [trialModalOpen, setTrialModalOpen] = useState(false);

  const navigate = useNavigate();
  const isMobile = window.innerWidth < 640;

  const handleLogin = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError('');

    try {
      await signInWithEmailAndPassword(auth, email, password);
      navigate('/dashboard');
    } catch (err: any) {
      console.error('ログインエラー:', err);
      setError('メールアドレスまたはパスワードが正しくありません');
    } finally {
      setLoading(false);
    }
  };

  const handleOpenTrialModal = () => {
    setTrialModalOpen(true);
  };

  const handleCloseTrialModal = () => {
    setTrialModalOpen(false);
  };

  return (
    <div className="container mx-auto h-screen flex items-center">
      <div className="flex w-full shadow-lg rounded-lg overflow-hidden">
        {/* 左側: イメージとキャッチコピー */}
        {!isMobile && (
          <div
            className="flex-1 bg-primary text-white p-8 flex flex-col justify-center"
          >
            <Building className="h-14 w-14 mb-4" />
            <h2 className="text-2xl font-bold mb-2">
              Startup Wellness
            </h2>
            <h3 className="text-xl mb-2">
              あなたのスタートアップの健康を分析
            </h3>
            <p className="mt-4">
              VASデータと財務データを組み合わせた高度な分析で、
              あなたのビジネスの健全性を可視化します。
            </p>

            <div className="mt-8">
              <p className="font-bold mb-2">
                主な機能:
              </p>
              <ul className="list-disc pl-5">
                <li>高度なデータ分析</li>
                <li>カスタムダッシュボード</li>
                <li>レポート自動生成</li>
                <li>AIによる洞察</li>
              </ul>
            </div>
          </div>
        )}

        {/* 右側: ログインフォーム */}
        <Card className="flex-1 p-8 flex flex-col justify-center">
          <div className="flex flex-col items-center mb-8">
            <Lock className="text-primary h-10 w-10 mb-2" />
            <h2 className="text-2xl font-bold">
              ログイン
            </h2>
          </div>

          {error && (
            <p className="text-destructive text-center mb-4">
              {error}
            </p>
          )}

          <form onSubmit={handleLogin} className="space-y-4">
            <div className="space-y-2">
              <Label htmlFor="email">メールアドレス</Label>
              <Input
                id="email"
                type="email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                required
                autoFocus
              />
            </div>

            <div className="space-y-2">
              <Label htmlFor="password">パスワード</Label>
              <Input
                id="password"
                type="password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                required
              />
            </div>

            <Button
              type="submit"
              className="w-full mt-6"
              disabled={loading}
            >
              {loading ? 'ログイン中...' : 'ログイン'}
            </Button>
          </form>

          <div className="mt-4 mb-6">
            <button
              onClick={() => navigate('/reset-password')}
              className="text-sm text-blue-600 hover:underline"
            >
              パスワードをお忘れですか？
            </button>
          </div>

          <div className="relative my-4">
            <Separator />
            <span className="absolute inset-x-0 top-1/2 transform -translate-y-1/2 text-center">
              <span className="bg-white px-2 text-muted-foreground text-sm">または</span>
            </span>
          </div>

          {/* トライアルCTA */}
          <div className="space-y-4 mt-4">
            <Button
              variant="outline"
              className="w-full py-6"
              onClick={handleOpenTrialModal}
            >
              <Star className="mr-2 h-4 w-4" />
              無料トライアルを試す
            </Button>
            <p className="text-sm text-muted-foreground text-center">
              アカウントをお持ちでない場合は、無料トライアルをお試しください
            </p>
          </div>
        </Card>
      </div>

      {/* トライアル選択モーダル */}
      <TrialSelectionModal open={trialModalOpen} onClose={handleCloseTrialModal} />
    </div>
  );
};

export default LoginPage;