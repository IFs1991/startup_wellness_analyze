import { useState } from 'react';
import { useForm } from 'react-hook-form';
import { z } from 'zod';
import { zodResolver } from '@hookform/resolvers/zod';
import { StartupDataService } from '../api/services';
import { StartupData } from '../api/types';

// バリデーションスキーマの定義
const startupSchema = z.object({
  name: z.string()
    .min(1, '企業名は必須です')
    .max(100, '企業名は100文字以内で入力してください'),
  founded_date: z.string()
    .min(1, '設立日は必須です'),
  industry: z.string()
    .min(1, '業種は必須です'),
  location: z.string()
    .min(1, '所在地は必須です'),
  funding_stage: z.string()
    .min(1, '資金調達ステージは必須です'),
  total_funding: z.number()
    .min(0, '総調達額は0以上で入力してください')
    .or(z.string().transform(val => val === '' ? 0 : Number(val))),
  employees_count: z.number()
    .int('従業員数は整数で入力してください')
    .min(1, '従業員数は1人以上で入力してください')
    .or(z.string().transform(val => val === '' ? 0 : Number(val))),
  revenue: z.number()
    .min(0, '売上は0以上で入力してください')
    .optional()
    .or(z.string().transform(val => val === '' ? undefined : Number(val))),
  burn_rate: z.number()
    .min(0, 'バーンレートは0以上で入力してください')
    .optional()
    .or(z.string().transform(val => val === '' ? undefined : Number(val))),
  runway: z.number()
    .min(0, 'ランウェイは0以上で入力してください')
    .optional()
    .or(z.string().transform(val => val === '' ? undefined : Number(val))),
});

type StartupFormData = z.infer<typeof startupSchema>;

const AddStartupPage = () => {
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [submitError, setSubmitError] = useState<string | null>(null);
  const [submitSuccess, setSubmitSuccess] = useState(false);

  const {
    register,
    handleSubmit,
    reset,
    formState: { errors }
  } = useForm<StartupFormData>({
    resolver: zodResolver(startupSchema),
    defaultValues: {
      name: '',
      founded_date: '',
      industry: '',
      location: '',
      funding_stage: '',
      total_funding: 0,
      employees_count: 1,
      revenue: undefined,
      burn_rate: undefined,
      runway: undefined,
    }
  });

  const onSubmit = async (data: StartupFormData) => {
    setIsSubmitting(true);
    setSubmitError(null);
    setSubmitSuccess(false);

    try {
      // バックエンドAPIを呼び出し
      const response = await StartupDataService.createStartup(data as Partial<StartupData>);

      if (response.status === 'success') {
        setSubmitSuccess(true);
        reset(); // フォームをリセット
      } else {
        setSubmitError(response.message || 'エラーが発生しました。もう一度お試しください。');
      }
    } catch (error) {
      console.error('企業登録エラー:', error);
      setSubmitError('サーバーとの通信中にエラーが発生しました。');
    } finally {
      setIsSubmitting(false);
    }
  };

  const fundingStages = [
    { value: '', label: '選択してください' },
    { value: 'pre-seed', label: 'プレシード' },
    { value: 'seed', label: 'シード' },
    { value: 'series_a', label: 'シリーズA' },
    { value: 'series_b', label: 'シリーズB' },
    { value: 'series_c', label: 'シリーズC' },
    { value: 'series_d_plus', label: 'シリーズD以上' },
    { value: 'ipo', label: 'IPO' },
  ];

  return (
    <div className="container mx-auto px-4 py-8">
      <h1 className="text-2xl font-bold mb-6">新規企業登録</h1>

      {submitSuccess && (
        <div className="bg-green-100 border border-green-400 text-green-700 px-4 py-3 rounded mb-4">
          企業が正常に登録されました。
        </div>
      )}

      {submitError && (
        <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded mb-4">
          {submitError}
        </div>
      )}

      <form onSubmit={handleSubmit(onSubmit)} className="space-y-6">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {/* 企業名 */}
          <div>
            <label htmlFor="name" className="block text-sm font-medium text-gray-700 mb-1">
              企業名 <span className="text-red-500">*</span>
            </label>
            <input
              id="name"
              type="text"
              {...register('name')}
              className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-primary focus:border-primary"
            />
            {errors.name && (
              <p className="mt-1 text-sm text-red-600">{errors.name.message}</p>
            )}
          </div>

          {/* 設立日 */}
          <div>
            <label htmlFor="founded_date" className="block text-sm font-medium text-gray-700 mb-1">
              設立日 <span className="text-red-500">*</span>
            </label>
            <input
              id="founded_date"
              type="date"
              {...register('founded_date')}
              className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-primary focus:border-primary"
            />
            {errors.founded_date && (
              <p className="mt-1 text-sm text-red-600">{errors.founded_date.message}</p>
            )}
          </div>

          {/* 業種 */}
          <div>
            <label htmlFor="industry" className="block text-sm font-medium text-gray-700 mb-1">
              業種 <span className="text-red-500">*</span>
            </label>
            <input
              id="industry"
              type="text"
              {...register('industry')}
              className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-primary focus:border-primary"
            />
            {errors.industry && (
              <p className="mt-1 text-sm text-red-600">{errors.industry.message}</p>
            )}
          </div>

          {/* 所在地 */}
          <div>
            <label htmlFor="location" className="block text-sm font-medium text-gray-700 mb-1">
              所在地 <span className="text-red-500">*</span>
            </label>
            <input
              id="location"
              type="text"
              {...register('location')}
              className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-primary focus:border-primary"
            />
            {errors.location && (
              <p className="mt-1 text-sm text-red-600">{errors.location.message}</p>
            )}
          </div>

          {/* 資金調達ステージ */}
          <div>
            <label htmlFor="funding_stage" className="block text-sm font-medium text-gray-700 mb-1">
              資金調達ステージ <span className="text-red-500">*</span>
            </label>
            <select
              id="funding_stage"
              {...register('funding_stage')}
              className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-primary focus:border-primary"
            >
              {fundingStages.map((stage) => (
                <option key={stage.value} value={stage.value}>
                  {stage.label}
                </option>
              ))}
            </select>
            {errors.funding_stage && (
              <p className="mt-1 text-sm text-red-600">{errors.funding_stage.message}</p>
            )}
          </div>

          {/* 総調達額 */}
          <div>
            <label htmlFor="total_funding" className="block text-sm font-medium text-gray-700 mb-1">
              総調達額（円）<span className="text-red-500">*</span>
            </label>
            <input
              id="total_funding"
              type="number"
              min="0"
              {...register('total_funding')}
              className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-primary focus:border-primary"
            />
            {errors.total_funding && (
              <p className="mt-1 text-sm text-red-600">{errors.total_funding.message}</p>
            )}
          </div>

          {/* 従業員数 */}
          <div>
            <label htmlFor="employees_count" className="block text-sm font-medium text-gray-700 mb-1">
              従業員数 <span className="text-red-500">*</span>
            </label>
            <input
              id="employees_count"
              type="number"
              min="1"
              {...register('employees_count')}
              className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-primary focus:border-primary"
            />
            {errors.employees_count && (
              <p className="mt-1 text-sm text-red-600">{errors.employees_count.message}</p>
            )}
          </div>

          {/* 売上（任意） */}
          <div>
            <label htmlFor="revenue" className="block text-sm font-medium text-gray-700 mb-1">
              売上（円）
            </label>
            <input
              id="revenue"
              type="number"
              min="0"
              {...register('revenue')}
              className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-primary focus:border-primary"
            />
            {errors.revenue && (
              <p className="mt-1 text-sm text-red-600">{errors.revenue.message}</p>
            )}
          </div>

          {/* バーンレート（任意） */}
          <div>
            <label htmlFor="burn_rate" className="block text-sm font-medium text-gray-700 mb-1">
              バーンレート（円/月）
            </label>
            <input
              id="burn_rate"
              type="number"
              min="0"
              {...register('burn_rate')}
              className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-primary focus:border-primary"
            />
            {errors.burn_rate && (
              <p className="mt-1 text-sm text-red-600">{errors.burn_rate.message}</p>
            )}
          </div>

          {/* ランウェイ（任意） */}
          <div>
            <label htmlFor="runway" className="block text-sm font-medium text-gray-700 mb-1">
              ランウェイ（月）
            </label>
            <input
              id="runway"
              type="number"
              min="0"
              {...register('runway')}
              className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-primary focus:border-primary"
            />
            {errors.runway && (
              <p className="mt-1 text-sm text-red-600">{errors.runway.message}</p>
            )}
          </div>
        </div>

        <div className="flex justify-end space-x-4">
          <button
            type="button"
            onClick={() => reset()}
            className="px-4 py-2 border border-gray-300 rounded-md shadow-sm text-sm font-medium text-gray-700 bg-white hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-primary"
            disabled={isSubmitting}
          >
            リセット
          </button>
          <button
            type="submit"
            className="px-4 py-2 border border-transparent rounded-md shadow-sm text-sm font-medium text-white bg-blue-600 hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500"
            disabled={isSubmitting}
          >
            {isSubmitting ? '登録中...' : '企業を登録する'}
          </button>
        </div>
      </form>
    </div>
  );
};

export default AddStartupPage;