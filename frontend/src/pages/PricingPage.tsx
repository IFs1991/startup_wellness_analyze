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
import { AlertCircle, Check, Loader2 } from 'lucide-react';
import { useNavigate } from 'react-router-dom';
import { useAuth } from '../hooks/useAuth';
import { useSubscription, SubscriptionPlan } from '../hooks/useSubscription';

const PricingPage: React.FC = () => {
  const { user } = useAuth();
  const navigate = useNavigate();
  const [annualBilling, setAnnualBilling] = useState(false);

  const {
    availablePlans,
    loading: subscriptionLoading,
    error: subscriptionError,
    changePlan,
  } = useSubscription();

  const loading = subscriptionLoading;
  const error = subscriptionError?.message;

  const handleBillingToggle = () => {
    setAnnualBilling((prev) => !prev);
  };

  const handleSelectPlan = async (planId: string) => {
    if (!user) {
      navigate('/login', { state: { returnUrl: '/pricing', selectedPlan: planId } });
      return;
    }

    try {
      await changePlan(planId);
      navigate('/subscription');
    } catch (err) {
      console.error("Plan selection error caught in page:", err);
    }
  };

  const calculateDiscountedPrice = (price: number): number => {
    return Math.floor(price * 0.8);
  };

  const renderPlans = () => {
    if (!availablePlans || availablePlans.length === 0) {
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
          {availablePlans.map((plan: SubscriptionPlan) => {
            const isRecommended = plan.name.toLowerCase().includes('pro');
            const monthlyPrice = plan.price / 100;
            const yearlyPrice = calculateDiscountedPrice(monthlyPrice * 12);
            const displayPrice = annualBilling ? yearlyPrice : monthlyPrice;

            return (
              <Card
                key={plan.id}
                className={`relative overflow-visible transition-all hover:-translate-y-2 hover:shadow-lg ${
                  isRecommended
                    ? 'border-2 border-primary shadow-lg'
                    : 'border'
                }`}
              >
                {isRecommended && (
                  <Badge
                    variant="default"
                    className="absolute -top-3 left-1/2 -translate-x-1/2 bg-primary text-primary-foreground"
                  >
                    おすすめ
                  </Badge>
                )}
                <CardHeader className="pt-8">
                  <CardTitle className="text-2xl font-bold mb-2">{plan.name}</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="mb-6">
                    <span className="text-4xl font-bold">¥{displayPrice.toLocaleString()}</span>
                    <span className="text-muted-foreground">/{annualBilling ? '年' : '月'}</span>
                  </div>
                  <Separator className="mb-6" />
                  <ul className="space-y-3">
                    {plan.features.map((feature, index) => (
                      <li key={index} className="flex items-center gap-2">
                        <Check className="h-4 w-4 text-green-500" />
                        <span className="text-sm">{feature}</span>
                      </li>
                    ))}
                    <li className="flex items-center gap-2">
                      <Check className="h-4 w-4 text-green-500" />
                      <span className="text-sm">最大企業数: {plan.maxCompanies}</span>
                    </li>
                    <li className="flex items-center gap-2">
                      <Check className="h-4 w-4 text-green-500" />
                      <span className="text-sm">最大従業員数: {plan.maxEmployees}</span>
                    </li>
                  </ul>
                </CardContent>
                <CardFooter>
                  <Button
                    onClick={() => handleSelectPlan(plan.id)}
                    className="w-full"
                    variant={isRecommended ? 'default' : 'outline'}
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
    <div className="container mx-auto py-12 px-4">
      {loading && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex justify-center items-center z-50">
          <Loader2 className="h-16 w-16 animate-spin text-white" />
        </div>
      )}
      {error && (
        <Alert variant="destructive" className="mb-8">
          <AlertCircle className="h-4 w-4" />
          <AlertTitle>エラー</AlertTitle>
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      )}

      {renderPlans()}
    </div>
  );
};

export default PricingPage;