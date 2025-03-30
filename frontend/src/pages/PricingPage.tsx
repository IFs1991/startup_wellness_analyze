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
import { Switch } from '@/components/ui/switch';
import { Label } from '@/components/ui/label';
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';
import { AlertCircle, Check, Timer, TrendingUp, Tag } from 'lucide-react';
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
      setError(null);
    } catch (err) {
      setError('カスタム価格情報の取得に失敗しました。');
      console.error('Pricing fetch error:', err);
    } finally {
      setLoading(false);
    }
  };

  const fetchStandardPricing = async () => {
    try {
      setLoading(true);
      const response = await api.get('/pricing/standard');
      setPricingData(response.data);
      setError(null);
    } catch (err) {
      setError('価格情報の取得に失敗しました。');
      console.error('Pricing fetch error:', err);
    } finally {
      setLoading(false);
    }
  };

  const handleBillingToggle = () => {
    setAnnualBilling((prev) => !prev);
  };

  const handleSelectPlan = async (planId: string) => {
    if (!user) {
      // ログインしていない場合はログイン画面にリダイレクト
      navigate('/login', { state: { returnUrl: '/pricing', selectedPlan: planId } });
      return;
    }

    try {
      setLoading(true);
      // 選択されたプランに基づいて課金ページにリダイレクト
      const response = await api.post('/billing/checkout-session', {
        plan_id: planId,
        billing_interval: annualBilling ? 'year' : 'month'
      });

      // 支払いページへリダイレクト
      window.location.href = response.data.url;
    } catch (err) {
      setError('決済の処理中にエラーが発生しました。');
      console.error('Checkout error:', err);
      setLoading(false);
    }
  };

  const calculateDiscountedPrice = (price: number): number => {
    return Math.floor(price * 0.8); // 20%割引を適用
  };

  const renderSpecialOffer = () => {
    if (!pricingData?.special_offer) return null;

    const offer = pricingData.special_offer;
    const expiryDate = new Date(offer.expires_at);

    return (
      <div className="mb-8 mt-4">
        <Alert variant="default" className="bg-amber-50 border-amber-200">
          <Tag className="h-4 w-4 text-amber-500" />
          <AlertTitle className="font-bold text-amber-800">
            スペシャルオファー: {offer.value} OFF
          </AlertTitle>
          <AlertDescription className="text-amber-700">
            {offer.duration}の間、すべてのプランに{offer.value}割引が適用されます。
            <span className="font-semibold block mt-1">
              コード：{offer.code} | 有効期限：{expiryDate.toLocaleDateString()}
            </span>
          </AlertDescription>
        </Alert>
      </div>
    );
  };

  // ユーザーのトライアル状態を表示
  const renderTrialNotice = () => {
    if (!pricingData?.trial_status || pricingData.trial_status.days_remaining <= 0) return null;

    return (
      <div className="mb-8">
        <Alert variant="default" className="bg-blue-50 border-blue-200">
          <Timer className="h-4 w-4 text-blue-500" />
          <AlertTitle className="font-bold text-blue-800">
            無料トライアル期間中です
          </AlertTitle>
          <AlertDescription className="text-blue-700">
            あと{pricingData.trial_status.days_remaining}日間の無料トライアルが残っています。
            <span className="font-semibold block mt-1">
              終了日: {new Date(pricingData.trial_status.end_date).toLocaleDateString()}
            </span>
          </AlertDescription>
        </Alert>
      </div>
    );
  };

  // ROI予測の表示
  const renderRoiProjection = () => {
    if (!pricingData?.roi_projection) return null;

    const roi = pricingData.roi_projection;

    return (
      <Card className="mb-8 border-green-100 bg-green-50">
        <CardHeader>
          <div className="flex items-center gap-2">
            <TrendingUp className="h-5 w-5 text-green-600" />
            <CardTitle className="text-green-800">ROI予測分析</CardTitle>
          </div>
          <CardDescription className="text-green-700">
            当社のサービスを利用することで期待できる投資対効果（ROI）の予測です
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="p-4 bg-white rounded-lg border border-green-100">
              <div className="text-sm text-green-700 mb-1">予測される月間節約時間</div>
              <div className="text-2xl font-bold text-green-800">{roi.time_saved} 時間</div>
            </div>
            <div className="p-4 bg-white rounded-lg border border-green-100">
              <div className="text-sm text-green-700 mb-1">予測される年間コスト削減</div>
              <div className="text-2xl font-bold text-green-800">{roi.cost_saved.toLocaleString()} 円</div>
            </div>
            <div className="p-4 bg-white rounded-lg border border-green-100">
              <div className="text-sm text-green-700 mb-1">予測される従業員満足度向上</div>
              <div className="text-2xl font-bold text-green-800">+{roi.satisfaction_increase}%</div>
            </div>
          </div>
        </CardContent>
      </Card>
    );
  };

  const renderPlans = () => {
    if (!pricingData?.plans || pricingData.plans.length === 0) {
      return (
        <div className="flex justify-center my-12">
          <div className="text-muted-foreground">利用可能なプランが見つかりません</div>
        </div>
      );
    }

    return (
      <>
        <div className="mb-8 flex flex-col sm:flex-row items-center justify-between">
          <h2 className="text-3xl font-bold mb-4 sm:mb-0">ご利用プラン</h2>
          <div className="flex items-center gap-2">
            <Label htmlFor="billing-toggle" className={!annualBilling ? "font-bold" : ""}>月払い</Label>
            <Switch
              id="billing-toggle"
              checked={annualBilling}
              onCheckedChange={handleBillingToggle}
            />
            <div className="flex items-center gap-2">
              <Label htmlFor="billing-toggle" className={annualBilling ? "font-bold" : ""}>年払い</Label>
              <Badge variant="outline" className="bg-green-50 text-green-700 border-green-200">
                20% OFF
              </Badge>
            </div>
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
          {pricingData.plans.map((plan) => {
            const isRecommended = pricingData.recommended_plan === plan.id || plan.is_recommended;
            const price = annualBilling
              ? calculateDiscountedPrice(plan.price * 12)
              : plan.price;

            return (
              <Card
                key={plan.id}
                className={`relative overflow-visible transition-all hover:-translate-y-2 hover:shadow-lg ${
                  isRecommended ? 'border-primary shadow-md' : 'border-border'
                }`}
              >
                {isRecommended && (
                  <Badge
                    className="absolute -top-3 left-1/2 transform -translate-x-1/2 bg-primary text-primary-foreground"
                  >
                    おすすめ
                  </Badge>
                )}

                <CardHeader>
                  <CardTitle>{plan.name}</CardTitle>
                  <CardDescription>{plan.description}</CardDescription>
                </CardHeader>

                <CardContent>
                  <div className="mb-4">
                    <div className="flex items-baseline gap-1">
                      <span className="text-3xl font-bold">
                        {plan.currency}
                        {price.toLocaleString()}
                      </span>
                      <span className="text-muted-foreground">
                        / {annualBilling ? '年' : plan.interval}
                      </span>
                    </div>
                    {annualBilling && (
                      <p className="text-green-600 text-sm mt-1">
                        月あたり {Math.floor(price / 12).toLocaleString()} 円（20% OFF）
                      </p>
                    )}
                  </div>

                  <Separator className="my-4" />

                  <div className="space-y-2 mt-4">
                    {plan.features.map((feature, idx) => (
                      <div key={idx} className="flex gap-2 items-start">
                        <Check className="h-4 w-4 text-green-500 mt-1 shrink-0" />
                        <span className="text-sm">{feature}</span>
                      </div>
                    ))}
                  </div>
                </CardContent>

                <CardFooter className="pt-0">
                  <Button
                    onClick={() => handleSelectPlan(plan.id)}
                    className="w-full"
                    variant={isRecommended ? "default" : "outline"}
                    disabled={loading}
                  >
                    {loading ? '処理中...' : 'プランを選択'}
                  </Button>
                </CardFooter>
              </Card>
            );
          })}
        </div>
      </>
    );
  };

  return (
    <div className="container mx-auto px-4">
      <div className="my-8">
        <div className="text-center mb-10">
          <h1 className="text-4xl font-bold mb-4">スタートアップウェルネス分析プラン</h1>
          <p className="text-muted-foreground text-lg max-w-2xl mx-auto">
            企業のウェルネスを分析し、生産性と従業員満足度を向上させるための最適なプランをお選びください
          </p>
        </div>

        {error && (
          <Alert variant="destructive" className="mb-8">
            <AlertCircle className="h-4 w-4" />
            <AlertTitle>エラー</AlertTitle>
            <AlertDescription>{error}</AlertDescription>
          </Alert>
        )}

        {loading && !pricingData ? (
          <div className="flex justify-center my-12">
            <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary"></div>
          </div>
        ) : (
          <>
            {renderTrialNotice()}
            {renderSpecialOffer()}
            {renderRoiProjection()}
            {renderPlans()}
          </>
        )}

        <div className="mt-16 text-center">
          <h3 className="text-xl font-semibold mb-2">お問い合わせ</h3>
          <p className="text-muted-foreground mb-4">
            カスタムプランや企業向けプランについては、お気軽にお問い合わせください。
          </p>
          <Button variant="outline">お問い合わせ</Button>
        </div>
      </div>
    </div>
  );
};

export default PricingPage;