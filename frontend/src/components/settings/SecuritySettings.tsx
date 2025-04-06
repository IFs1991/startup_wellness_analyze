import React from 'react';
import {
  Card, CardContent, CardHeader, Typography, Box,
  FormControlLabel, Switch, FormGroup, FormLabel
} from '@mui/material';
import { UserSettings } from '@/hooks/useSettings'; // UserSettings型をインポート

// Propsの型定義
interface SecuritySettingsProps {
  settings: UserSettings['privacy'];
  updateSettings: (updatedSettings: Partial<UserSettings>) => Promise<void>;
}

export const SecuritySettings: React.FC<SecuritySettingsProps> = ({ settings, updateSettings }) => {

  // スイッチの状態変更ハンドラ
  const handleChange = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const { name, checked } = event.target;
    try {
      // privacy オブジェクト全体を更新する形で updateSettings を呼び出す
      await updateSettings({
        privacy: {
          ...settings, // 既存の設定を維持
          [name]: checked // 変更された設定を適用
        }
      });
      // TODO: 成功時のフィードバック
    } catch (error) {
      console.error("プライバシー設定の更新に失敗しました:", error);
      // TODO: エラー時のフィードバック
    }
  };

  return (
    <Card>
      <CardHeader title="プライバシー設定" />
      <CardContent>
        <Typography variant="body1" paragraph>
          データ共有とマーケティングに関する設定です。
        </Typography>
        <Box sx={{ mt: 2 }}>
          <FormGroup>
            <FormLabel component="legend" sx={{ mb: 1 }}>データ共有</FormLabel>
            <FormControlLabel
              control={
                <Switch
                  checked={settings.shareAnalytics}
                  onChange={handleChange}
                  name="shareAnalytics"
                />
              }
              label="匿名の分析データを共有する"
            />
            <FormControlLabel
              control={
                <Switch
                  checked={settings.shareUsageData}
                  onChange={handleChange}
                  name="shareUsageData"
                />
              }
              label="匿名の利用状況データを共有する"
            />
            <FormLabel component="legend" sx={{ mt: 3, mb: 1 }}>マーケティング</FormLabel>
            <FormControlLabel
              control={
                <Switch
                  checked={settings.marketingEmails}
                  onChange={handleChange}
                  name="marketingEmails"
                />
              }
              label="サービスに関するお知らせやマーケティングメールを受け取る"
            />
          </FormGroup>
        </Box>
      </CardContent>
    </Card>
  );
};