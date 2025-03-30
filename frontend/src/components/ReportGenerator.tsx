import React, { useState } from 'react';
import { Button, Card, Form, Select, Spin, message, Alert } from 'antd';
import { DownloadOutlined, FileTextOutlined } from '@ant-design/icons';
import { generateReport } from '../services/api';

const { Option } = Select;

interface ReportGeneratorProps {
  companyId: string;
  companyData: any;
}

const ReportGenerator: React.FC<ReportGeneratorProps> = ({ companyId, companyData }) => {
  const [loading, setLoading] = useState(false);
  const [reportUrl, setReportUrl] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [selectedTemplate, setSelectedTemplate] = useState('standard');
  const [selectedFormat, setSelectedFormat] = useState('pdf');
  const [selectedPeriod, setSelectedPeriod] = useState('last_quarter');

  // 含めるセクションの選択（デフォルトですべて選択）
  const [selectedSections, setSelectedSections] = useState([
    'wellness_score',
    'financial_metrics',
    'wellness_metrics',
    'correlation_analysis',
    'recommendations'
  ]);

  const handleGenerateReport = async () => {
    setLoading(true);
    setError(null);
    setReportUrl(null);

    try {
      const result = await generateReport({
        template_id: selectedTemplate,
        company_data: companyData,
        period: selectedPeriod,
        include_sections: selectedSections,
        format: selectedFormat
      });

      if (result.success) {
        setReportUrl(result.report_url);
        message.success('レポートが正常に生成されました');
      } else {
        setError(result.error || 'レポートの生成に失敗しました');
        message.error('レポートの生成に失敗しました');
      }
    } catch (err) {
      setError('APIリクエスト中にエラーが発生しました');
      message.error('APIリクエスト中にエラーが発生しました');
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <Card title="企業レポート生成" className="report-generator">
      <Form layout="vertical">
        <Form.Item label="レポートテンプレート">
          <Select
            value={selectedTemplate}
            onChange={setSelectedTemplate}
            style={{ width: '100%' }}
          >
            <Option value="standard">標準レポート</Option>
            <Option value="executive">エグゼクティブサマリー</Option>
            <Option value="detailed">詳細分析レポート</Option>
          </Select>
        </Form.Item>

        <Form.Item label="分析期間">
          <Select
            value={selectedPeriod}
            onChange={setSelectedPeriod}
            style={{ width: '100%' }}
          >
            <Option value="last_month">直近1ヶ月</Option>
            <Option value="last_quarter">直近四半期</Option>
            <Option value="last_year">直近1年</Option>
            <Option value="all_time">全期間</Option>
          </Select>
        </Form.Item>

        <Form.Item label="含めるセクション">
          <Select
            mode="multiple"
            value={selectedSections}
            onChange={setSelectedSections}
            style={{ width: '100%' }}
          >
            <Option value="wellness_score">ウェルネススコア概要</Option>
            <Option value="financial_metrics">財務指標</Option>
            <Option value="wellness_metrics">健康指標</Option>
            <Option value="correlation_analysis">相関分析</Option>
            <Option value="recommendations">改善提案</Option>
          </Select>
        </Form.Item>

        <Form.Item label="出力形式">
          <Select
            value={selectedFormat}
            onChange={setSelectedFormat}
            style={{ width: '100%' }}
          >
            <Option value="pdf">PDF</Option>
            <Option value="html">HTML</Option>
          </Select>
        </Form.Item>

        <Form.Item>
          <Button
            type="primary"
            icon={<FileTextOutlined />}
            onClick={handleGenerateReport}
            loading={loading}
            block
          >
            レポート生成
          </Button>
        </Form.Item>

        {error && (
          <Alert
            message="エラー"
            description={error}
            type="error"
            showIcon
            style={{ marginBottom: '16px' }}
          />
        )}

        {reportUrl && (
          <Button
            type="default"
            icon={<DownloadOutlined />}
            href={reportUrl}
            target="_blank"
            block
          >
            レポートをダウンロード
          </Button>
        )}
      </Form>

      {loading && (
        <div className="loading-container">
          <Spin tip="レポートを生成中..." />
          <p>企業データと分析結果を基にレポートを作成しています。しばらくお待ちください...</p>
        </div>
      )}
    </Card>
  );
};

export default ReportGenerator;