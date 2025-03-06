import React, { useState, useEffect } from 'react';
import {
  Container,
  Typography,
  Box,
  Grid,
  Card,
  CardContent,
  CardHeader,
  Button,
  Chip,
  Divider,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  Paper,
  Switch,
  FormGroup,
  FormControlLabel,
  CircularProgress,
  Alert,
  useTheme,
  useMediaQuery,
} from '@mui/material';
import {
  Check as CheckIcon,
  CheckCircle as CheckCircleIcon,
  StarBorder as StarBorderIcon,
  Star as StarIcon,
  ArrowForward as ArrowForwardIcon,
  Timer as TimerIcon,
  TrendingUp as TrendingUpIcon,
  LocalOffer as LocalOfferIcon,
} from '@mui/icons-material';
import { useNavigate } from 'react-router-dom';
import { useAuth } from '../hooks/useAuth';
import api from '../services/api';

interface Plan {
  id: string;
  name: string;
  price: number;
  currency: string;
  interval: string;
  description: string;
  features: string[];
  is_recommended?: boolean;
}

interface SpecialOffer {
  type: string;
  value: string;
  duration: string;
  code: string;
  expires_at: string;
}

interface PricingData {
  plans: Plan[];
  recommended_plan?: string;
  user_usage_summary?: any;
  roi_projection?: any;
  special_offer?: SpecialOffer;
  trial_status?: {
    end_date: string;
    days_remaining: number;
  };
}

const PricingPage: React.FC = () => {
  const { user } = useAuth();
  const navigate = useNavigate();
  const theme = useTheme();
  const isMobile = useMediaQuery(theme.breakpoints.down('sm'));
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [pricingData, setPricingData] = useState<PricingData | null>(null);
  const [annualBilling, setAnnualBilling] = useState(false);

  useEffect(() => {
    if (user) {
      fetchPersonalizedPricing();
    } else {
      fetchStandardPricing();
    }
  }, [user]);

  const fetchPersonalizedPricing = async () => {
    try {
      setLoading(true);
      const response = await api.get('/pricing/personalized');
      setPricingData(response.data);
    } catch (err: any) {
      console.error('パーソナライズされた価格データの取得エラー:', err);
      setError('価格データの取得中にエラーが発生しました');
      // エラー時はデフォルトの価格データにフォールバック
      fetchStandardPricing();
    } finally {
      setLoading(false);
    }
  };

  const fetchStandardPricing = async () => {
    try {
      setLoading(true);
      const response = await api.get('/subscriptions/plans');
      setPricingData({
        plans: response.data,
      });
    } catch (err: any) {
      console.error('価格データの取得エラー:', err);
      setError('価格データの取得中にエラーが発生しました');
    } finally {
      setLoading(false);
    }
  };

  const handleBillingToggle = () => {
    setAnnualBilling(!annualBilling);
  };

  const handleSelectPlan = async (planId: string) => {
    if (!user) {
      // 未ログインの場合はログインページにリダイレクト
      navigate('/login');
      return;
    }

    try {
      const response = await api.post('/subscriptions/checkout', {
        plan_id: planId
      });

      // Stripeのチェックアウトページにリダイレクト
      window.location.href = response.data.checkout_url;
    } catch (err: any) {
      console.error('チェックアウト作成エラー:', err);
      setError('チェックアウトの作成中にエラーが発生しました');
    }
  };

  const calculateDiscountedPrice = (price: number): number => {
    return annualBilling ? Math.round(price * 0.8) : price;
  };

  const renderSpecialOffer = () => {
    if (!pricingData?.special_offer) return null;

    const offer = pricingData.special_offer;
    const expiresAt = new Date(offer.expires_at);
    const now = new Date();
    const daysRemaining = Math.ceil((expiresAt.getTime() - now.getTime()) / (1000 * 60 * 60 * 24));

    return (
      <Paper
        sx={{
          p: 3,
          mb: 4,
          bgcolor: '#fff9c4',
          borderRadius: 2,
          position: 'relative',
          overflow: 'hidden',
        }}
      >
        <Box
          sx={{
            position: 'absolute',
            top: 0,
            right: 0,
            bgcolor: '#fbc02d',
            color: 'white',
            py: 0.5,
            px: 2,
            transform: 'rotate(45deg) translate(20%, -70%)',
            transformOrigin: 'top right',
            boxShadow: 1,
          }}
        >
          <Typography variant="subtitle2">あなた限定</Typography>
        </Box>

        <Grid container spacing={2} alignItems="center">
          <Grid item xs={12} md={2}>
            <LocalOfferIcon sx={{ fontSize: 60, color: '#f57c00' }} />
          </Grid>
          <Grid item xs={12} md={7}>
            <Typography variant="h5" gutterBottom fontWeight="bold">
              特別オファー: {offer.value} OFF
            </Typography>
            <Typography variant="body1" gutterBottom>
              {offer.duration}の間、{offer.value}割引が適用されます
            </Typography>
            <Box sx={{ display: 'flex', alignItems: 'center', mt: 1 }}>
              <TimerIcon sx={{ mr: 1, color: 'text.secondary' }} />
              <Typography variant="body2" color="text.secondary">
                あと{daysRemaining}日間の期間限定オファー
              </Typography>
            </Box>
          </Grid>
          <Grid item xs={12} md={3} sx={{ textAlign: { xs: 'center', md: 'right' } }}>
            <Chip
              label={`コード: ${offer.code}`}
              sx={{ mb: 1, fontWeight: 'bold' }}
              color="primary"
            />
            <Typography variant="body2" sx={{ mb: 1 }}>
              チェックアウト時に自動適用
            </Typography>
          </Grid>
        </Grid>
      </Paper>
    );
  };

  const renderTrialNotice = () => {
    if (!pricingData?.trial_status || pricingData.trial_status.days_remaining <= 0) return null;

    return (
      <Alert
        severity="info"
        sx={{ mb: 4 }}
        icon={<TimerIcon />}
      >
        <Typography variant="body1">
          トライアル期間はあと{pricingData.trial_status.days_remaining}日間です。
          トライアル終了日: {new Date(pricingData.trial_status.end_date).toLocaleDateString('ja-JP')}
        </Typography>
      </Alert>
    );
  };

  const renderRoiProjection = () => {
    if (!pricingData?.roi_projection) return null;

    const roi = pricingData.roi_projection;

    return (
      <Paper sx={{ p: 3, mb: 4, bgcolor: '#e3f2fd', borderRadius: 2 }}>
        <Typography variant="h6" gutterBottom>
          <TrendingUpIcon sx={{ mr: 1, verticalAlign: 'middle' }} />
          導入効果予測
        </Typography>
        <Grid container spacing={3} sx={{ mt: 1 }}>
          <Grid item xs={12} md={4}>
            <Box sx={{ textAlign: 'center', p: 1 }}>
              <Typography variant="h5" color="primary" fontWeight="bold">
                {roi.time_saved_monthly} 時間/月
              </Typography>
              <Typography variant="body2">作業時間の削減</Typography>
            </Box>
          </Grid>
          <Grid item xs={12} md={4}>
            <Box sx={{ textAlign: 'center', p: 1 }}>
              <Typography variant="h5" color="primary" fontWeight="bold">
                {roi.cost_saved_monthly.toLocaleString()} 円/月
              </Typography>
              <Typography variant="body2">コスト削減</Typography>
            </Box>
          </Grid>
          <Grid item xs={12} md={4}>
            <Box sx={{ textAlign: 'center', p: 1 }}>
              <Typography variant="h5" color="primary" fontWeight="bold">
                {roi.roi_percentage}%
              </Typography>
              <Typography variant="body2">投資対効果 (ROI)</Typography>
            </Box>
          </Grid>
        </Grid>
      </Paper>
    );
  };

  const renderPlans = () => {
    if (!pricingData || !pricingData.plans) return null;

    // 推奨プランのIDを取得
    const recommendedPlanId = pricingData.recommended_plan;

    return (
      <>
        <Box sx={{ display: 'flex', justifyContent: 'center', mb: 4 }}>
          <FormGroup>
            <FormControlLabel
              control={
                <Switch checked={annualBilling} onChange={handleBillingToggle} color="primary" />
              }
              label={
                <Box sx={{ display: 'flex', alignItems: 'center' }}>
                  <Typography>月額</Typography>
                  <Box
                    sx={{
                      mx: 1,
                      px: 1,
                      py: 0.5,
                      bgcolor: 'success.light',
                      color: 'success.contrastText',
                      borderRadius: 1,
                      fontSize: '0.75rem',
                      fontWeight: 'bold',
                    }}
                  >
                    20%お得
                  </Box>
                  <Typography>年額</Typography>
                </Box>
              }
              labelPlacement="end"
            />
          </FormGroup>
        </Box>

        <Grid container spacing={3} alignItems="stretch">
          {pricingData.plans.map((plan) => {
            const isRecommended = plan.id === recommendedPlanId || plan.is_recommended;
            const cardBorder = isRecommended ? `2px solid ${theme.palette.primary.main}` : undefined;
            const price = calculateDiscountedPrice(plan.price);

            return (
              <Grid item xs={12} md={4} key={plan.id}>
                <Card
                  sx={{
                    height: '100%',
                    display: 'flex',
                    flexDirection: 'column',
                    border: cardBorder,
                    position: 'relative',
                    ...(isRecommended && {
                      boxShadow: 6,
                      transform: { md: 'scale(1.05)' },
                      zIndex: 1
                    })
                  }}
                >
                  {isRecommended && (
                    <Box
                      sx={{
                        position: 'absolute',
                        top: 12,
                        right: 0,
                        bgcolor: 'primary.main',
                        color: 'white',
                        py: 0.5,
                        px: 2,
                        borderTopLeftRadius: 16,
                        borderBottomLeftRadius: 16,
                      }}
                    >
                      <Typography variant="caption" fontWeight="bold">おすすめ</Typography>
                    </Box>
                  )}

                  <CardHeader
                    title={plan.name}
                    subheader={plan.description}
                    titleTypographyProps={{ align: 'center', fontWeight: 'bold' }}
                    subheaderTypographyProps={{ align: 'center' }}
                    sx={{ bgcolor: isRecommended ? 'primary.light' : 'grey.100', color: 'text.primary' }}
                  />
                  <CardContent sx={{ flexGrow: 1, display: 'flex', flexDirection: 'column' }}>
                    <Box sx={{ textAlign: 'center', mb: 2 }}>
                      <Typography variant="h4" component="span" fontWeight="bold">
                        {price.toLocaleString()}
                      </Typography>
                      <Typography variant="h6" component="span" color="text.secondary">
                        円/{annualBilling ? '年' : '月'}
                      </Typography>

                      {annualBilling && (
                        <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
                          月額換算: {Math.round(price / 12).toLocaleString()}円/月
                        </Typography>
                      )}
                    </Box>

                    <Divider sx={{ my: 2 }} />

                    <List sx={{ flexGrow: 1 }}>
                      {plan.features.map((feature, index) => (
                        <ListItem key={index} sx={{ py: 0.5 }}>
                          <ListItemIcon sx={{ minWidth: 36 }}>
                            <CheckIcon color="success" />
                          </ListItemIcon>
                          <ListItemText primary={feature} />
                        </ListItem>
                      ))}
                    </List>

                    <Button
                      fullWidth
                      variant={isRecommended ? "contained" : "outlined"}
                      color="primary"
                      sx={{ mt: 2 }}
                      onClick={() => handleSelectPlan(plan.id)}
                      endIcon={<ArrowForwardIcon />}
                    >
                      このプランを選択
                    </Button>
                  </CardContent>
                </Card>
              </Grid>
            );
          })}
        </Grid>
      </>
    );
  };

  if (loading) {
    return (
      <Container maxWidth="lg" sx={{ py: 8 }}>
        <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', minHeight: '50vh' }}>
          <CircularProgress />
        </Box>
      </Container>
    );
  }

  return (
    <Container maxWidth="lg" sx={{ py: 8 }}>
      <Box sx={{ textAlign: 'center', mb: 6 }}>
        <Typography variant="h3" component="h1" gutterBottom fontWeight="bold">
          価格プラン
        </Typography>
        <Typography variant="h6" color="text.secondary" sx={{ maxWidth: 800, mx: 'auto' }}>
          あなたのビジネスに最適なプランをお選びください。
          いつでもアップグレードやダウングレードが可能です。
        </Typography>
      </Box>

      {renderTrialNotice()}
      {renderSpecialOffer()}
      {renderRoiProjection()}

      {error && (
        <Alert severity="error" sx={{ mb: 4 }}>
          {error}
        </Alert>
      )}

      {renderPlans()}

      <Box sx={{ mt: 8, textAlign: 'center' }}>
        <Typography variant="h6" gutterBottom>
          ご不明点がございましたら、お気軽にお問い合わせください
        </Typography>
        <Button
          variant="outlined"
          color="primary"
          size="large"
          sx={{ mt: 2 }}
          href="/contact"
        >
          お問い合わせ
        </Button>
      </Box>
    </Container>
  );
};

export default PricingPage;