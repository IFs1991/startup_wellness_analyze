import React from 'react';
import { Card, CardContent, CardHeader, Typography, Box } from '@mui/material';

export const SecuritySettings: React.FC = () => {
  return (
    <Card>
      <CardHeader title="セキュリティ設定" />
      <CardContent>
        <Typography variant="body1" paragraph>
          アカウントのセキュリティ設定を管理します。
        </Typography>
        <Box sx={{ mt: 2 }}>
          <Typography variant="body2" color="text.secondary">
            セキュリティ設定の内容はまだ実装されていません。
          </Typography>
        </Box>
      </CardContent>
    </Card>
  );
};