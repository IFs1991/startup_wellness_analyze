import React from 'react';
import { Card, CardContent, CardHeader, Typography, Box } from '@mui/material';

export const NotificationSettings: React.FC = () => {
  return (
    <Card>
      <CardHeader title="通知設定" />
      <CardContent>
        <Typography variant="body1" paragraph>
          通知の受信設定を管理します。
        </Typography>
        <Box sx={{ mt: 2 }}>
          <Typography variant="body2" color="text.secondary">
            通知設定の内容はまだ実装されていません。
          </Typography>
        </Box>
      </CardContent>
    </Card>
  );
};