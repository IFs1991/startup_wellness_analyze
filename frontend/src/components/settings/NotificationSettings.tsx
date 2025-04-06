import React from 'react';
import {
  Card, CardContent, CardHeader, Typography, Box,
  FormControlLabel, Switch, FormGroup, FormLabel
} from '@mui/material';
import { UserSettings } from '@/hooks/useSettings'; // UserSettings型をインポート

// Propsの型定義
interface NotificationSettingsProps {
  settings: UserSettings['notifications'];
  updateSettings: (updatedSettings: Partial<UserSettings>) => Promise<void>;
}

export const NotificationSettings: React.FC<NotificationSettingsProps> = ({ settings, updateSettings }) => {

  // スイッチの状態変更ハンドラ
  const handleChange = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const { name, checked } = event.target;
    try {
      // notifications オブジェクト全体を更新する形で updateSettings を呼び出す
      await updateSettings({
        notifications: {
          ...settings, // 既存の設定を維持
          [name]: checked // 変更された設定を適用
        }
      });
      // TODO: 成功時のフィードバック（例：Toast表示）
    } catch (error) {
      console.error("通知設定の更新に失敗しました:", error);
      // TODO: エラー時のフィードバック
    }
  };

  return (
    <Card>
      <CardHeader title="通知設定" />
      <CardContent>
        <Typography variant="body1" paragraph>
          受け取る通知の種類を選択してください。
        </Typography>
        <Box sx={{ mt: 2 }}>
          <FormGroup>
            <FormLabel component="legend" sx={{ mb: 1 }}>通知チャネル</FormLabel>
            <FormControlLabel
              control={
                <Switch
                  checked={settings.email}
                  onChange={handleChange}
                  name="email"
                />
              }
              label="メール通知"
            />
            <FormControlLabel
              control={
                <Switch
                  checked={settings.push}
                  onChange={handleChange}
                  name="push"
                  disabled // プッシュ通知は未実装などの理由で無効化されている場合
                />
              }
              label="プッシュ通知 (未対応)"
            />
            <FormLabel component="legend" sx={{ mt: 3, mb: 1 }}>通知内容</FormLabel>
            <FormControlLabel
              control={
                <Switch
                  checked={settings.reportGeneration}
                  onChange={handleChange}
                  name="reportGeneration"
                />
              }
              label="レポート生成完了通知"
            />
            <FormControlLabel
              control={
                <Switch
                  checked={settings.dataUpdates}
                  onChange={handleChange}
                  name="dataUpdates"
                />
              }
              label="データ更新/インサイト通知"
            />
          </FormGroup>
        </Box>
      </CardContent>
    </Card>
  );
};