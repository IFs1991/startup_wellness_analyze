import React, { useState } from 'react';
import { X } from 'lucide-react';

interface Company {
  id: string;
  name: string;
  industry: string;
  stage: string;
  location: string;
  employeesCount: number;
  wellnessScore: number;
  growthRate: number;
}

interface AddCompanyDialogProps {
  open: boolean;
  onClose: () => void;
  onAddCompany: (company: Omit<Company, 'id' | 'wellnessScore'>) => void;
}

const industries = ['SaaS', 'ヘルスケア', 'クリーンテック', 'フィンテック', 'デザイン', 'ソフトウェア', 'ハードウェア', 'その他'];
const stages = ['シード', 'シリーズA', 'シリーズB', 'シリーズC以降'];
const locations = ['東京', '大阪', '名古屋', '福岡', '京都', '北海道', 'その他'];

interface FormErrors {
  name?: string;
  industry?: string;
  stage?: string;
  location?: string;
  employeesCount?: string;
  growthRate?: string;
}

const AddCompanyDialog: React.FC<AddCompanyDialogProps> = ({ open, onClose, onAddCompany }) => {
  const [formData, setFormData] = useState({
    name: '',
    industry: '',
    stage: '',
    location: '',
    employeesCount: '',
    growthRate: ''
  });
  const [errors, setErrors] = useState<FormErrors>({});

  if (!open) return null;

  const handleChange = (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) => {
    const { name, value } = e.target;
    setFormData(prev => ({ ...prev, [name]: value }));

    // エラーを消去
    if (errors[name as keyof FormErrors]) {
      setErrors(prev => ({ ...prev, [name]: undefined }));
    }
  };

  const validateForm = (): boolean => {
    const newErrors: FormErrors = {};

    if (!formData.name.trim()) {
      newErrors.name = '企業名を入力してください';
    }

    if (!formData.industry) {
      newErrors.industry = '業界を選択してください';
    }

    if (!formData.stage) {
      newErrors.stage = 'ステージを選択してください';
    }

    if (!formData.location) {
      newErrors.location = '所在地を選択してください';
    }

    if (!formData.employeesCount) {
      newErrors.employeesCount = '従業員数を入力してください';
    } else if (isNaN(Number(formData.employeesCount)) || Number(formData.employeesCount) <= 0) {
      newErrors.employeesCount = '有効な従業員数を入力してください';
    }

    if (!formData.growthRate) {
      newErrors.growthRate = '成長率を入力してください';
    } else if (isNaN(Number(formData.growthRate))) {
      newErrors.growthRate = '有効な成長率を入力してください';
    }

    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();

    if (validateForm()) {
      onAddCompany({
        name: formData.name,
        industry: formData.industry,
        stage: formData.stage,
        location: formData.location,
        employeesCount: Number(formData.employeesCount),
        growthRate: Number(formData.growthRate)
      });
    }
  };

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
      <div className="bg-gray-900 rounded-lg p-6 w-full max-w-md">
        <div className="flex justify-between items-center mb-4">
          <h2 className="text-xl font-bold">企業を追加</h2>
          <button
            onClick={onClose}
            className="text-gray-400 hover:text-white"
          >
            <X className="h-5 w-5" />
          </button>
        </div>

        <form onSubmit={handleSubmit}>
          <div className="space-y-4">
            {/* 企業名 */}
            <div>
              <label htmlFor="name" className="block text-sm font-medium text-gray-300 mb-1">
                企業名 <span className="text-red-500">*</span>
              </label>
              <input
                type="text"
                id="name"
                name="name"
                value={formData.name}
                onChange={handleChange}
                className="w-full bg-gray-800 border border-gray-700 rounded-lg py-2 px-3 text-white focus:outline-none focus:ring-2 focus:ring-blue-500"
                placeholder="テックスタート株式会社"
              />
              {errors.name && <p className="text-red-500 text-xs mt-1">{errors.name}</p>}
            </div>

            {/* 業界 */}
            <div>
              <label htmlFor="industry" className="block text-sm font-medium text-gray-300 mb-1">
                業界 <span className="text-red-500">*</span>
              </label>
              <select
                id="industry"
                name="industry"
                value={formData.industry}
                onChange={handleChange}
                className="w-full bg-gray-800 border border-gray-700 rounded-lg py-2 px-3 text-white focus:outline-none focus:ring-2 focus:ring-blue-500"
              >
                <option value="">選択してください</option>
                {industries.map(industry => (
                  <option key={industry} value={industry}>{industry}</option>
                ))}
              </select>
              {errors.industry && <p className="text-red-500 text-xs mt-1">{errors.industry}</p>}
            </div>

            {/* ステージ */}
            <div>
              <label htmlFor="stage" className="block text-sm font-medium text-gray-300 mb-1">
                ステージ <span className="text-red-500">*</span>
              </label>
              <select
                id="stage"
                name="stage"
                value={formData.stage}
                onChange={handleChange}
                className="w-full bg-gray-800 border border-gray-700 rounded-lg py-2 px-3 text-white focus:outline-none focus:ring-2 focus:ring-blue-500"
              >
                <option value="">選択してください</option>
                {stages.map(stage => (
                  <option key={stage} value={stage}>{stage}</option>
                ))}
              </select>
              {errors.stage && <p className="text-red-500 text-xs mt-1">{errors.stage}</p>}
            </div>

            {/* 所在地 */}
            <div>
              <label htmlFor="location" className="block text-sm font-medium text-gray-300 mb-1">
                所在地 <span className="text-red-500">*</span>
              </label>
              <select
                id="location"
                name="location"
                value={formData.location}
                onChange={handleChange}
                className="w-full bg-gray-800 border border-gray-700 rounded-lg py-2 px-3 text-white focus:outline-none focus:ring-2 focus:ring-blue-500"
              >
                <option value="">選択してください</option>
                {locations.map(location => (
                  <option key={location} value={location}>{location}</option>
                ))}
              </select>
              {errors.location && <p className="text-red-500 text-xs mt-1">{errors.location}</p>}
            </div>

            {/* 従業員数 */}
            <div>
              <label htmlFor="employeesCount" className="block text-sm font-medium text-gray-300 mb-1">
                従業員数 <span className="text-red-500">*</span>
              </label>
              <input
                type="number"
                id="employeesCount"
                name="employeesCount"
                value={formData.employeesCount}
                onChange={handleChange}
                className="w-full bg-gray-800 border border-gray-700 rounded-lg py-2 px-3 text-white focus:outline-none focus:ring-2 focus:ring-blue-500"
                min="1"
                placeholder="10"
              />
              {errors.employeesCount && <p className="text-red-500 text-xs mt-1">{errors.employeesCount}</p>}
            </div>

            {/* 成長率 */}
            <div>
              <label htmlFor="growthRate" className="block text-sm font-medium text-gray-300 mb-1">
                成長率（%） <span className="text-red-500">*</span>
              </label>
              <input
                type="number"
                id="growthRate"
                name="growthRate"
                value={formData.growthRate}
                onChange={handleChange}
                className="w-full bg-gray-800 border border-gray-700 rounded-lg py-2 px-3 text-white focus:outline-none focus:ring-2 focus:ring-blue-500"
                placeholder="15"
              />
              {errors.growthRate && <p className="text-red-500 text-xs mt-1">{errors.growthRate}</p>}
            </div>

            <div className="flex justify-end space-x-3 mt-6">
              <button
                type="button"
                onClick={onClose}
                className="px-4 py-2 bg-gray-800 hover:bg-gray-700 text-white rounded-lg transition duration-150"
              >
                キャンセル
              </button>
              <button
                type="submit"
                className="px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg transition duration-150"
              >
                追加
              </button>
            </div>
          </div>
        </form>
      </div>
    </div>
  );
};

export default AddCompanyDialog;