import React, { useState, useEffect } from 'react';
import {
  Container,
  Typography,
  Box,
  Card,
  CardContent,
  Button,
  Chip,
  Divider,
  Grid,
  Paper,
  CircularProgress,
  Alert,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogContentText,
  DialogActions,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  useTheme,
} from '@mui/material';
import {
  CheckCircle as CheckCircleIcon,
  CreditCard as CreditCardIcon,
  Info as InfoIcon,
  Warning as WarningIcon,
  EventAvailable as EventAvailableIcon,
  EventBusy as EventBusyIcon,
  Receipt as ReceiptIcon,
  AccessTime as AccessTimeIcon,
} from '@mui/icons-material';
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
  const theme = useTheme();
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

        // 請求書履歴を取得
        const invoiceResponse = await api.get('/subscriptions/invoices');
        setInvoices(invoiceResponse.data.invoices);
      }
    } catch (err: any) {
      console.error('サブスクリプションデータの取得エラー:', err);
      setError('サブスクリプション情報の取得中にエラーが発生しました。');
    } finally {
      setLoading(false);
    }
  };

  const handleOpenCancelDialog = () => {
    setOpenCancelDialog(true);
    setCancelError(null);
    setCancelSuccess(false);
  };

  const handleCloseCancelDialog = () => {
    setOpenCancelDialog(false);
  };

  const handleCancelSubscription = async () => {
    if (!subscription) return;

    try {
      setCancelLoading(true);
      setCancelError(null);

      // サブスクリプションをキャンセル（更新停止）
      await api.post('/subscriptions/cancel', {
        subscription_id: subscription.id,
        at_period_end: true
      });

      setCancelSuccess(true);

      // サブスクリプション情報を再取得
      await fetchSubscriptionData();
    } catch (err: any) {
      console.error('サブスクリプションキャンセルエラー:', err);
      setCancelError('サブスクリプションのキャンセル処理中にエラーが発生しました。');
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
    if (!subscription) return;

    try {
      setReactivateLoading(true);

      // サブスクリプションを再開
      await api.post('/subscriptions/reactivate', {
        subscription_id: subscription.id
      });

      // サブスクリプション情報を再取得
      await fetchSubscriptionData();
      setOpenReactivateDialog(false);
    } catch (err: any) {
      console.error('サブスクリプション再開エラー:', err);
      setError('サブスクリプションの再開処理中にエラーが発生しました。');
    } finally {
      setReactivateLoading(false);
    }
  };

  const formatDate = (dateString: string) => {
    return format(new Date(dateString), 'yyyy年MM月dd日', { locale: ja });
  };

  const getPlanName = (planType: string) => {
    const planNames: {[key: string]: string} = {
      'free': '無料プラン',
      'basic': 'ベーシックプラン',
      'premium': 'プレミアムプラン',
      'business': 'ビジネスプラン',
      'enterprise': 'エンタープライズプラン'
    };
    return planNames[planType] || planType;
  };

  const getStatusChip = () => {
    if (!subscription) return null;

    if (subscription.status === 'trialing') {
      return <Chip label="トライアル中" color="info" icon={<InfoIcon />} />;
    } else if (subscription.status === 'active') {
      if (subscription.cancel_at) {
        return <Chip label="更新停止予定" color="warning" icon={<EventBusyIcon />} />;
      }
      return <Chip label="有効" color="success" icon={<CheckCircleIcon />} />;
    } else if (subscription.status === 'canceled') {
      return <Chip label="キャンセル済み" color="error" icon={<EventBusyIcon />} />;
    } else if (subscription.status === 'free') {
      return <Chip label="無料プラン" color="default" />;
    }
    return <Chip label={subscription.status} />;
  };

  const renderSubscriptionDetails = () => {
    if (!subscription) return null;

    return (
      <Card sx={{ mb: 4 }}>
        <CardContent>
          <Grid container spacing={2}>
            <Grid item xs={12} md={6}>
              <Typography variant="h6" gutterBottom>
                現在のプラン
              </Typography>
              <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                <Typography variant="h5" component="span" fontWeight="bold" sx={{ mr: 2 }}>
                  {getPlanName(subscription.plan_type)}
                </Typography>
                {getStatusChip()}
              </Box>

              {subscription.trial_end && new Date(subscription.trial_end) > new Date() && (
                <Alert severity="info" sx={{ mb: 2 }}>
                  <Typography variant="body2">
                    トライアル期間は {formatDate(subscription.trial_end)} に終了します。
                    {subscription.status === 'trialing' && subscription.plan_type !== 'free' && (
                      '以降は自動的に課金が開始されます。'
                    )}
                  </Typography>
                </Alert>
              )}

              <Box sx={{ mt: 2 }}>
                <Typography variant="body2" color="text.secondary" gutterBottom>
                  <AccessTimeIcon fontSize="small" sx={{ mr: 1, verticalAlign: 'middle' }} />
                  現在の期間: {formatDate(subscription.current_period_end)}まで
                </Typography>

                {subscription.cancel_at && (
                  <Typography variant="body2" color="error" gutterBottom>
                    <EventBusyIcon fontSize="small" sx={{ mr: 1, verticalAlign: 'middle' }} />
                    {formatDate(subscription.cancel_at)}に自動更新が停止されます
                  </Typography>
                )}
              </Box>
            </Grid>

            <Grid item xs={12} md={6}>
              <Typography variant="h6" gutterBottom>
                プラン操作
              </Typography>

              {subscription.status === 'active' && !subscription.cancel_at && (
                <Button
                  variant="outlined"
                  color="warning"
                  onClick={handleOpenCancelDialog}
                  sx={{ mb: 2 }}
                  fullWidth
                >
                  自動更新を停止
                </Button>
              )}

              {subscription.status === 'active' && subscription.cancel_at && (
                <Button
                  variant="outlined"
                  color="primary"
                  onClick={handleOpenReactivateDialog}
                  sx={{ mb: 2 }}
                  fullWidth
                >
                  自動更新を再開
                </Button>
              )}

              {subscription.status !== 'free' && (
                <Button
                  variant="outlined"
                  color="info"
                  href="/pricing"
                  sx={{ mb: 2 }}
                  fullWidth
                >
                  プランを変更
                </Button>
              )}

              {subscription.status === 'free' && (
                <Button
                  variant="contained"
                  color="primary"
                  href="/pricing"
                  fullWidth
                >
                  有料プランにアップグレード
                </Button>
              )}
            </Grid>
          </Grid>
        </CardContent>
      </Card>
    );
  };

  const renderBillingInfo = () => {
    if (!billingInfo) return null;

    return (
      <Card sx={{ mb: 4 }}>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            お支払い情報
          </Typography>

          <Box sx={{ display: 'flex', alignItems: 'center', mt: 2 }}>
            <CreditCardIcon sx={{ mr: 1 }} color="action" />
            <Typography>
              {billingInfo.brand} •••• {billingInfo.last4} (有効期限: {billingInfo.exp_month}/{billingInfo.exp_year})
            </Typography>
          </Box>

          <Box sx={{ mt: 3 }}>
            <Button
              variant="outlined"
              size="small"
              href="/update-payment"
            >
              支払い方法を更新
            </Button>
          </Box>
        </CardContent>
      </Card>
    );
  };

  const renderInvoices = () => {
    if (!invoices || invoices.length === 0) return null;

    return (
      <Card>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            請求履歴
          </Typography>
          <List>
            {invoices.map((invoice) => (
              <React.Fragment key={invoice.id}>
                <ListItem>
                  <ListItemIcon>
                    <ReceiptIcon />
                  </ListItemIcon>
                  <ListItemText
                    primary={`${invoice.amount_paid / 100} ${invoice.currency.toUpperCase()}`}
                    secondary={`${format(new Date(invoice.period_start * 1000), 'yyyy年MM月dd日')} - ${format(new Date(invoice.period_end * 1000), 'yyyy年MM月dd日')}`}
                  />
                  <Button
                    size="small"
                    variant="outlined"
                    href={`/api/subscriptions/invoice-pdf/${invoice.id}`}
                    target="_blank"
                  >
                    領収書
                  </Button>
                </ListItem>
                <Divider component="li" />
              </React.Fragment>
            ))}
          </List>
        </CardContent>
      </Card>
    );
  };

  // キャンセル確認ダイアログ
  const renderCancelDialog = () => (
    <Dialog
      open={openCancelDialog}
      onClose={handleCloseCancelDialog}
    >
      <DialogTitle>サブスクリプションの自動更新を停止しますか？</DialogTitle>
      <DialogContent>
        <DialogContentText>
          自動更新を停止すると、現在の請求期間（{subscription ? formatDate(subscription.current_period_end) : ''}まで）は引き続きサービスをご利用いただけますが、その後は自動的に更新されません。
        </DialogContentText>
        {cancelError && (
          <Alert severity="error" sx={{ mt: 2 }}>
            {cancelError}
          </Alert>
        )}
        {cancelSuccess && (
          <Alert severity="success" sx={{ mt: 2 }}>
            自動更新の停止が完了しました。現在の期間終了後、サブスクリプションは更新されません。
          </Alert>
        )}
      </DialogContent>
      <DialogActions>
        <Button onClick={handleCloseCancelDialog} color="primary">
          キャンセル
        </Button>
        <Button
          onClick={handleCancelSubscription}
          color="warning"
          disabled={cancelLoading || cancelSuccess}
        >
          {cancelLoading ? <CircularProgress size={24} /> : '自動更新を停止する'}
        </Button>
      </DialogActions>
    </Dialog>
  );

  // 再アクティベートダイアログ
  const renderReactivateDialog = () => (
    <Dialog
      open={openReactivateDialog}
      onClose={handleCloseReactivateDialog}
    >
      <DialogTitle>サブスクリプションの自動更新を再開しますか？</DialogTitle>
      <DialogContent>
        <DialogContentText>
          自動更新を再開すると、現在の請求期間終了後も自動的にサブスクリプションが更新されます。
        </DialogContentText>
      </DialogContent>
      <DialogActions>
        <Button onClick={handleCloseReactivateDialog} color="primary">
          キャンセル
        </Button>
        <Button
          onClick={handleReactivateSubscription}
          color="primary"
          disabled={reactivateLoading}
        >
          {reactivateLoading ? <CircularProgress size={24} /> : '自動更新を再開する'}
        </Button>
      </DialogActions>
    </Dialog>
  );

  if (loading) {
    return (
      <Container maxWidth="md" sx={{ py: 4 }}>
        <Box sx={{ display: 'flex', justifyContent: 'center', py: 8 }}>
          <CircularProgress />
        </Box>
      </Container>
    );
  }

  if (error) {
    return (
      <Container maxWidth="md" sx={{ py: 4 }}>
        <Alert severity="error" sx={{ mb: 3 }}>
          {error}
        </Alert>
        <Button variant="outlined" onClick={fetchSubscriptionData}>
          再読み込み
        </Button>
      </Container>
    );
  }

  return (
    <Container maxWidth="md" sx={{ py: 4 }}>
      <Typography variant="h4" component="h1" gutterBottom>
        サブスクリプション管理
      </Typography>

      {renderSubscriptionDetails()}
      {renderBillingInfo()}
      {renderInvoices()}
      {renderCancelDialog()}
      {renderReactivateDialog()}
    </Container>
  );
};

export default SubscriptionPage;