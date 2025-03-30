import React, { useState, useEffect } from 'react';
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
  CardDescription,
  CardFooter
} from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Separator } from '@/components/ui/separator';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogFooter,
} from '@/components/ui/dialog';
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table';
import {
  CheckCircle,
  CreditCard,
  AlertCircle,
  XCircle,
  FileText,
  Clock,
  ArrowRight,
} from 'lucide-react';
import { format } from 'date-fns';
import { ja } from 'date-fns/locale';
import api from '../services/api';
import { useAuth } from '../hooks/useAuth';

interface Subscription {
  id: string;
  status: string;
  plan_type: string;
  trial_end: string | null;
  current_period_end: string;
  cancel_at: string | null;
  canceled_at: string | null;
}

interface BillingInfo {
  last4: string;
  brand: string;
  exp_month: number;
  exp_year: number;
}

interface Invoice {
  id: string;
  amount_paid: number;
  currency: string;
  status: string;
  created: number;
  period_start: number;
  period_end: number;
}

const SubscriptionPage: React.FC = () => {
  const { user } = useAuth();
  const [subscription, setSubscription] = useState<Subscription | null>(null);
  const [billingInfo, setBillingInfo] = useState<BillingInfo | null>(null);
  const [invoices, setInvoices] = useState<Invoice[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [openCancelDialog, setOpenCancelDialog] = useState(false);
  const [cancelLoading, setCancelLoading] = useState(false);
  const [cancelError, setCancelError] = useState<string | null>(null);
  const [cancelSuccess, setCancelSuccess] = useState(false);
  const [openReactivateDialog, setOpenReactivateDialog] = useState(false);
  const [reactivateLoading, setReactivateLoading] = useState(false);

  useEffect(() => {
    if (user) {
      fetchSubscriptionData();
    }
  }, [user]);

  const fetchSubscriptionData = async () => {
    try {
      setLoading(true);
      setError(null);

      // サブスクリプション情報を取得
      const subResponse = await api.get('/subscriptions/status');
      setSubscription(subResponse.data.subscription);

      // クレジットカード情報を取得（存在する場合）
      if (subResponse.data.subscription && subResponse.data.subscription.status !== 'free') {
        const paymentResponse = await api.get('/subscriptions/payment-method');
        setBillingInfo(paymentResponse.data.payment_method);

        // 請求書情報を取得
        const invoicesResponse = await api.get('/subscriptions/invoices');
        setInvoices(invoicesResponse.data.invoices);
      }
    } catch (err) {
      console.error('サブスクリプションデータ取得エラー:', err);
      setError('サブスクリプション情報の取得に失敗しました。後でもう一度お試しください。');
    } finally {
      setLoading(false);
    }
  };

  const handleOpenCancelDialog = () => {
    setCancelError(null);
    setCancelSuccess(false);
    setOpenCancelDialog(true);
  };

  const handleCloseCancelDialog = () => {
    setOpenCancelDialog(false);
  };

  const handleCancelSubscription = async () => {
    try {
      setCancelLoading(true);
      setCancelError(null);

      await api.post('/subscriptions/cancel', {
        subscription_id: subscription?.id
      });

      // サブスクリプションの状態を更新
      setCancelSuccess(true);

      // 最新のデータを取得
      await fetchSubscriptionData();

      // 成功メッセージを表示した後にダイアログを閉じる
      setTimeout(() => {
        handleCloseCancelDialog();
      }, 2000);
    } catch (err) {
      console.error('サブスクリプション解約エラー:', err);
      setCancelError('サブスクリプションの解約に失敗しました。後でもう一度お試しください。');
    } finally {
      setCancelLoading(false);
    }
  };

  const handleOpenReactivateDialog = () => {
    setOpenReactivateDialog(true);
  };

  const handleCloseReactivateDialog = () => {
    setOpenReactivateDialog(false);
  };

  const handleReactivateSubscription = async () => {
    try {
      setReactivateLoading(true);

      await api.post('/subscriptions/reactivate', {
        subscription_id: subscription?.id
      });

      // サブスクリプションの状態を更新
      await fetchSubscriptionData();

      // ダイアログを閉じる
      handleCloseReactivateDialog();
    } catch (err) {
      console.error('サブスクリプション再開エラー:', err);
      setError('サブスクリプションの再開に失敗しました。後でもう一度お試しください。');
    } finally {
      setReactivateLoading(false);
    }
  };

  const formatDate = (dateString: string) => {
    return format(new Date(dateString), 'yyyy年MM月dd日', { locale: ja });
  };

  const getPlanName = (planType: string) => {
    switch (planType) {
      case 'basic':
        return 'ベーシック';
      case 'professional':
        return 'プロフェッショナル';
      case 'enterprise':
        return 'エンタープライズ';
      default:
        return '無料プラン';
    }
  };

  const getStatusChip = () => {
    if (!subscription) return null;

    if (subscription.status === 'active') {
      return <Badge className="bg-green-500">有効</Badge>;
    } else if (subscription.status === 'trialing') {
      return <Badge className="bg-blue-500">トライアル中</Badge>;
    } else if (subscription.status === 'canceled' && subscription.cancel_at) {
      return <Badge className="bg-amber-500">期間終了まで有効</Badge>;
    } else if (subscription.status === 'canceled') {
      return <Badge className="bg-red-500">キャンセル済み</Badge>;
    } else if (subscription.status === 'free') {
      return <Badge variant="outline">無料プラン</Badge>;
    }

    return <Badge variant="outline">{subscription.status}</Badge>;
  };

  const renderSubscriptionDetails = () => {
    if (!subscription) return null;

    return (
      <Card className="mb-8">
        <CardHeader>
          <div className="flex items-center justify-between">
            <div>
              <CardTitle>サブスクリプション</CardTitle>
              <CardDescription>現在のサブスクリプションとその状態</CardDescription>
            </div>
            {getStatusChip()}
          </div>
        </CardHeader>
        <CardContent>
          <div className="space-y-6">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div>
                <h3 className="text-sm font-medium text-muted-foreground mb-1">プラン</h3>
                <p className="text-lg font-medium">{getPlanName(subscription.plan_type)}</p>
              </div>

              <div>
                <h3 className="text-sm font-medium text-muted-foreground mb-1">ステータス</h3>
                <div className="flex items-center gap-2">
                  {subscription.status === 'active' && (
                    <>
                      <CheckCircle className="h-5 w-5 text-green-500" />
                      <span className="text-lg font-medium">有効</span>
                    </>
                  )}
                  {subscription.status === 'trialing' && (
                    <>
                      <Clock className="h-5 w-5 text-blue-500" />
                      <span className="text-lg font-medium">トライアル期間中</span>
                    </>
                  )}
                  {subscription.status === 'canceled' && (
                    <>
                      <XCircle className="h-5 w-5 text-red-500" />
                      <span className="text-lg font-medium">キャンセル済み</span>
                    </>
                  )}
                </div>
              </div>
            </div>

            <Separator />

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {subscription.trial_end && (
                <div>
                  <h3 className="text-sm font-medium text-muted-foreground mb-1">トライアル終了日</h3>
                  <div className="flex items-center gap-2">
                    <Clock className="h-5 w-5 text-blue-500" />
                    <span className="text-lg font-medium">{formatDate(subscription.trial_end)}</span>
                  </div>
                </div>
              )}

              <div>
                <h3 className="text-sm font-medium text-muted-foreground mb-1">現在の請求期間</h3>
                <div className="flex items-center gap-2">
                  <CreditCard className="h-5 w-5 text-primary" />
                  <span className="text-lg font-medium">{formatDate(subscription.current_period_end)}まで</span>
                </div>
              </div>
            </div>

            {subscription.cancel_at && (
              <>
                <Separator />
                <div>
                  <h3 className="text-sm font-medium text-muted-foreground mb-1">キャンセル有効日</h3>
                  <div className="flex items-center gap-2">
                    <XCircle className="h-5 w-5 text-red-500" />
                    <div>
                      <span className="text-lg font-medium">{formatDate(subscription.cancel_at)}</span>
                      <p className="text-sm text-muted-foreground mt-1">
                        この日付以降、サブスクリプションは無効になります。それまでは引き続きサービスをご利用いただけます。
                      </p>
                    </div>
                  </div>
                </div>
              </>
            )}
          </div>
        </CardContent>
        <CardFooter className="flex justify-end gap-2 pt-0">
          {/* アクティブなサブスクリプションの場合、キャンセルボタンを表示 */}
          {(subscription.status === 'active' || subscription.status === 'trialing') && !subscription.cancel_at && (
            <Button variant="outline" onClick={handleOpenCancelDialog}>
              サブスクリプションをキャンセル
            </Button>
          )}

          {/* キャンセル済みだが、まだ有効期間内の場合、再開ボタンを表示 */}
          {subscription.status === 'canceled' && subscription.cancel_at && new Date(subscription.cancel_at) > new Date() && (
            <Button onClick={handleOpenReactivateDialog}>
              サブスクリプションを再開
            </Button>
          )}

          {/* 無料プランの場合、アップグレードボタンを表示 */}
          {subscription.status === 'free' && (
            <Button asChild>
              <a href="/pricing">プランをアップグレード</a>
            </Button>
          )}
        </CardFooter>
      </Card>
    );
  };

  const renderBillingInfo = () => {
    if (!subscription || subscription.status === 'free' || !billingInfo) return null;

    return (
      <Card className="mb-8">
        <CardHeader>
          <CardTitle>支払い情報</CardTitle>
          <CardDescription>現在登録されている支払い方法</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            <div className="flex items-center gap-3">
              <CreditCard className="h-5 w-5 text-primary" />
              <div>
                <h3 className="font-medium">
                  {billingInfo.brand.charAt(0).toUpperCase() + billingInfo.brand.slice(1)}
                  &nbsp;&bull;&bull;&bull;&bull; {billingInfo.last4}
                </h3>
                <p className="text-sm text-muted-foreground">
                  有効期限: {billingInfo.exp_month}/{billingInfo.exp_year}
                </p>
              </div>
            </div>
          </div>
        </CardContent>
        <CardFooter className="pt-0">
          <Button variant="outline" asChild>
            <a href="/billing">支払い方法を更新</a>
          </Button>
        </CardFooter>
      </Card>
    );
  };

  const renderInvoices = () => {
    if (!subscription || subscription.status === 'free' || invoices.length === 0) {
      return null;
    }

    return (
      <Card className="mb-8">
        <CardHeader>
          <CardTitle>請求履歴</CardTitle>
          <CardDescription>過去の請求書</CardDescription>
        </CardHeader>
        <CardContent>
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>日付</TableHead>
                <TableHead>金額</TableHead>
                <TableHead>ステータス</TableHead>
                <TableHead className="text-right">アクション</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {invoices.map((invoice) => (
                <TableRow key={invoice.id}>
                  <TableCell>
                    {format(new Date(invoice.created * 1000), 'yyyy年MM月dd日', { locale: ja })}
                  </TableCell>
                  <TableCell>
                    {invoice.currency === 'jpy'
                      ? `¥${invoice.amount_paid.toLocaleString()}`
                      : `${invoice.amount_paid.toLocaleString()} ${invoice.currency.toUpperCase()}`}
                  </TableCell>
                  <TableCell>
                    {invoice.status === 'paid' && (
                      <Badge className="bg-green-500">支払い済み</Badge>
                    )}
                    {invoice.status === 'open' && (
                      <Badge variant="outline">未払い</Badge>
                    )}
                    {invoice.status === 'void' && (
                      <Badge variant="outline" className="bg-red-100 text-red-800">無効</Badge>
                    )}
                  </TableCell>
                  <TableCell className="text-right">
                    <Button size="sm" variant="ghost" asChild>
                      <a href={`/api/invoices/${invoice.id}`} target="_blank" rel="noopener noreferrer">
                        <FileText className="h-4 w-4 mr-1" />
                        <span>表示</span>
                      </a>
                    </Button>
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </CardContent>
      </Card>
    );
  };

  const renderCancelDialog = () => (
    <Dialog open={openCancelDialog} onOpenChange={setOpenCancelDialog}>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>サブスクリプションをキャンセルしますか？</DialogTitle>
          <DialogDescription>
            現在の請求期間の終了時にサブスクリプションはキャンセルされます。
            それまでは引き続きサービスをご利用いただけます。
          </DialogDescription>
        </DialogHeader>

        {cancelError && (
          <Alert variant="destructive">
            <AlertCircle className="h-4 w-4" />
            <AlertTitle>エラー</AlertTitle>
            <AlertDescription>{cancelError}</AlertDescription>
          </Alert>
        )}

        {cancelSuccess && (
          <Alert>
            <CheckCircle className="h-4 w-4" />
            <AlertTitle>成功</AlertTitle>
            <AlertDescription>
              サブスクリプションは正常にキャンセルされました。
              現在の請求期間の終了まで引き続きサービスをご利用いただけます。
            </AlertDescription>
          </Alert>
        )}

        <DialogFooter>
          <Button variant="outline" onClick={handleCloseCancelDialog} disabled={cancelLoading}>
            キャンセルしない
          </Button>
          <Button
            variant="destructive"
            onClick={handleCancelSubscription}
            disabled={cancelLoading || cancelSuccess}
          >
            {cancelLoading ? (
              <>
                <span className="animate-spin mr-2">&#9696;</span>
                処理中...
              </>
            ) : (
              'サブスクリプションを解約する'
            )}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );

  const renderReactivateDialog = () => (
    <Dialog open={openReactivateDialog} onOpenChange={setOpenReactivateDialog}>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>サブスクリプションを再開しますか？</DialogTitle>
          <DialogDescription>
            サブスクリプションを再開すると、予定されていたキャンセルはなくなり、
            次の請求日にお支払いが継続されます。
          </DialogDescription>
        </DialogHeader>

        <DialogFooter>
          <Button variant="outline" onClick={handleCloseReactivateDialog} disabled={reactivateLoading}>
            再開しない
          </Button>
          <Button
            onClick={handleReactivateSubscription}
            disabled={reactivateLoading}
          >
            {reactivateLoading ? (
              <>
                <span className="animate-spin mr-2">&#9696;</span>
                処理中...
              </>
            ) : (
              'サブスクリプションを再開する'
            )}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );

  if (loading) {
    return (
      <div className="container mx-auto px-4">
        <div className="my-8 flex justify-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary"></div>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="container mx-auto px-4">
        <div className="my-8">
          <Alert variant="destructive">
            <AlertCircle className="h-4 w-4" />
            <AlertTitle>エラー</AlertTitle>
            <AlertDescription>{error}</AlertDescription>
          </Alert>
          <div className="mt-4 text-center">
            <Button onClick={fetchSubscriptionData}>再試行</Button>
          </div>
        </div>
      </div>
    );
  }

  if (!subscription) {
    return (
      <div className="container mx-auto px-4">
        <div className="my-8">
          <Card>
            <CardHeader>
              <CardTitle>サブスクリプション情報</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-center py-8">
                <p className="text-muted-foreground mb-4">まだサブスクリプションがありません。</p>
                <Button asChild>
                  <a href="/pricing">
                    プランを見る
                    <ArrowRight className="ml-2 h-4 w-4" />
                  </a>
                </Button>
              </div>
            </CardContent>
          </Card>
        </div>
      </div>
    );
  }

  return (
    <div className="container mx-auto px-4">
      <div className="my-8">
        <div className="mb-6">
          <h1 className="text-3xl font-bold">サブスクリプション管理</h1>
          <p className="text-muted-foreground">
            サブスクリプション情報を確認・管理できます
          </p>
        </div>

        {renderSubscriptionDetails()}
        {renderBillingInfo()}
        {renderInvoices()}
        {renderCancelDialog()}
        {renderReactivateDialog()}
      </div>
    </div>
  );
};

export { SubscriptionPage };