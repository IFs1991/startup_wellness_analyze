import React, { useState } from 'react';
import { Link } from 'react-router-dom';
import {
  Box,
  Typography,
  Tabs,
  Tab,
  Card,
  CardContent,
  CardHeader,
  Container,
  Paper
} from '@mui/material';
import { SecuritySettings } from '@/components/settings/SecuritySettings';
import { NotificationSettings } from '@/components/settings/NotificationSettings';
import { UserManagement } from '@/components/settings/UserManagement';

interface TabPanelProps {
  children?: React.ReactNode;
  index: number;
  value: number;
}

function TabPanel(props: TabPanelProps) {
  const { children, value, index, ...other } = props;

  return (
    <Box
      role="tabpanel"
      hidden={value !== index}
      id={`settings-tabpanel-${index}`}
      aria-labelledby={`settings-tab-${index}`}
      {...other}
      sx={{ mt: 3 }}
    >
      {value === index && (
        <Box>
          {children}
        </Box>
      )}
    </Box>
  );
}

function a11yProps(index: number) {
  return {
    id: `settings-tab-${index}`,
    'aria-controls': `settings-tabpanel-${index}`,
  };
}

export const SettingsPage: React.FC = () => {
  const [tabValue, setTabValue] = useState(0);

  const handleTabChange = (event: React.SyntheticEvent, newValue: number) => {
    setTabValue(newValue);
  };

  return (
    <Container maxWidth="lg" sx={{ py: 4 }}>
      <Typography variant="h4" component="h1" gutterBottom sx={{ mb: 4 }}>
        設定
      </Typography>

      <Card sx={{ mb: 4 }}>
        <CardHeader title="アカウント設定" />
        <CardContent>
          <Typography variant="body1">
            アカウント設定からセキュリティ、通知、ユーザー管理、サブスクリプションなどを管理できます。
          </Typography>
        </CardContent>
      </Card>

      <Paper sx={{ mb: 3 }}>
        <Tabs
          value={tabValue}
          onChange={handleTabChange}
          aria-label="設定タブ"
          variant="scrollable"
          scrollButtons="auto"
        >
          <Tab label="セキュリティ" {...a11yProps(0)} />
          <Tab label="通知" {...a11yProps(1)} />
          <Tab label="ユーザー管理" {...a11yProps(2)} />
          <Tab label="サブスクリプション" {...a11yProps(3)} />
        </Tabs>
      </Paper>

      <TabPanel value={tabValue} index={0}>
        <SecuritySettings />
      </TabPanel>

      <TabPanel value={tabValue} index={1}>
        <NotificationSettings />
      </TabPanel>

      <TabPanel value={tabValue} index={2}>
        <UserManagement />
      </TabPanel>

      <TabPanel value={tabValue} index={3}>
        <Card>
          <CardHeader title="サブスクリプション情報" />
          <CardContent>
            <Typography variant="body1" paragraph>
              現在のサブスクリプションプランと契約状況を確認できます。
            </Typography>
            <Box sx={{ textAlign: 'center', my: 4 }}>
              <Link to="/subscription" style={{ textDecoration: 'none' }}>
                <Typography
                  variant="button"
                  sx={{
                    bgcolor: 'primary.main',
                    color: 'white',
                    py: 1.5,
                    px: 4,
                    borderRadius: 1,
                    boxShadow: 2,
                    '&:hover': {
                      bgcolor: 'primary.dark'
                    }
                  }}
                >
                  サブスクリプション管理ページへ
                </Typography>
              </Link>
            </Box>
          </CardContent>
        </Card>
      </TabPanel>
    </Container>
  );
};