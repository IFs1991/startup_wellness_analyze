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
  Paper,
  CircularProgress,
  Alert,
  TextField,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  Button,
  Stack,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  IconButton,
} from '@mui/material';
import AddIcon from '@mui/icons-material/Add';
import EditIcon from '@mui/icons-material/Edit';
import DeleteIcon from '@mui/icons-material/Delete';
import { useSettings } from '@/hooks/useSettings';
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

  const {
    userSettings,
    systemSettings,
    loading,
    error,
    updateUserSettings,
    resetSettings,
    refreshSettings
  } = useSettings();

  const [theme, setTheme] = useState(userSettings?.theme || 'light');
  const [language, setLanguage] = useState(userSettings?.language || 'ja');

  const handleTabChange = (_event: React.SyntheticEvent, newValue: number) => {
    setTabValue(newValue);
  };

  React.useEffect(() => {
    if (userSettings) {
      setTheme(userSettings.theme);
      setLanguage(userSettings.language);
    }
  }, [userSettings]);

  const handleGeneralSettingsSave = async () => {
    try {
      await updateUserSettings({ theme, language });
      console.log("General settings saved successfully");
    } catch (err) {
      console.error("Failed to save general settings:", err);
    }
  };

  if (loading) {
    return (
      <Container maxWidth="lg" sx={{ py: 4, display: 'flex', justifyContent: 'center', alignItems: 'center', minHeight: '80vh' }}>
        <CircularProgress />
        <Typography sx={{ ml: 2 }}>設定を読み込んでいます...</Typography>
      </Container>
    );
  }

  if (error || !userSettings) {
    return (
      <Container maxWidth="lg" sx={{ py: 4 }}>
        <Alert severity="error">
          設定の読み込みに失敗しました: {error?.message || 'データがありません'}
          <Button onClick={refreshSettings} size="small" sx={{ ml: 2 }}>再試行</Button>
        </Alert>
      </Container>
    );
  }

  return (
    <Container maxWidth="lg" sx={{ py: 4 }}>
      <Typography variant="h4" component="h1" gutterBottom sx={{ mb: 4 }}>
        設定
      </Typography>

      {systemSettings && (
        <Typography variant="caption" display="block" gutterBottom sx={{ mb: 2 }}>
          システムバージョン: {systemSettings.version}
          {systemSettings.maintenance && <span style={{ color: 'orange', marginLeft: '1em' }}>(メンテナンス中)</span>}
        </Typography>
      )}

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
          <Tab label="表示・一般" {...a11yProps(0)} />
          <Tab label="通知" {...a11yProps(1)} />
          <Tab label="セキュリティ" {...a11yProps(2)} />
          <Tab label="ユーザー管理" {...a11yProps(3)} />
          <Tab label="サブスクリプション" {...a11yProps(4)} />
        </Tabs>
      </Paper>

      <TabPanel value={tabValue} index={0}>
        <Card>
          <CardHeader title="表示・一般設定" />
          <CardContent>
            <Stack spacing={3}>
              <FormControl fullWidth>
                <InputLabel id="theme-select-label">テーマ</InputLabel>
                <Select
                  labelId="theme-select-label"
                  id="theme-select"
                  value={theme}
                  label="テーマ"
                  onChange={(e) => setTheme(e.target.value)}
                >
                  <MenuItem value={'light'}>ライト</MenuItem>
                  <MenuItem value={'dark'}>ダーク</MenuItem>
                </Select>
              </FormControl>
              <FormControl fullWidth>
                <InputLabel id="language-select-label">言語</InputLabel>
                <Select
                  labelId="language-select-label"
                  id="language-select"
                  value={language}
                  label="言語"
                  onChange={(e) => setLanguage(e.target.value)}
                >
                  <MenuItem value={'ja'}>日本語</MenuItem>
                  <MenuItem value={'en'}>English</MenuItem>
                </Select>
              </FormControl>
              <Box sx={{ textAlign: 'right' }}>
                <Button
                  variant="contained"
                  onClick={handleGeneralSettingsSave}
                  disabled={loading}
                >
                  変更を保存
                </Button>
              </Box>
            </Stack>
          </CardContent>
        </Card>
      </TabPanel>

      <TabPanel value={tabValue} index={1}>
        <NotificationSettings settings={userSettings.notifications} updateSettings={updateUserSettings} />
      </TabPanel>

      <TabPanel value={tabValue} index={2}>
        <SecuritySettings settings={userSettings.privacy} updateSettings={updateUserSettings} />
      </TabPanel>

      <TabPanel value={tabValue} index={3}>
        <UserManagement />
      </TabPanel>

      <TabPanel value={tabValue} index={4}>
        <Card>
          <CardHeader title="サブスクリプション情報" />
          <CardContent>
            <Typography variant="body1" paragraph>
              現在のサブスクリプションプランと契約状況を確認できます。
            </Typography>
            <Box sx={{ textAlign: 'center', my: 4 }}>
              <Link to="/subscription" style={{ textDecoration: 'none' }}>
                <Button variant="contained">
                  サブスクリプション管理ページへ
                </Button>
              </Link>
            </Box>
          </CardContent>
        </Card>
      </TabPanel>
    </Container>
  );
};