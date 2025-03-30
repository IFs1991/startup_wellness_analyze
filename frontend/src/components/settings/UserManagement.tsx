import React from 'react';
import { Card, CardContent, CardHeader, Typography, Box } from '@mui/material';

export const UserManagement: React.FC = () => {
  return (
    <Card>
      <CardHeader title="ユーザー管理" />
      <CardContent>
        <Typography variant="body1" paragraph>
          アカウントのユーザー管理設定を行います。
        </Typography>
        <Box sx={{ mt: 2 }}>
          <Typography variant="body2" color="text.secondary">
            ユーザー管理機能はまだ実装されていません。
          </Typography>
        </Box>
      </CardContent>
    </Card>
  );
};