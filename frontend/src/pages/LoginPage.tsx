import React, { useState } from 'react';
import { useNavigate, Link } from 'react-router-dom';
import { TextField, Button, Paper, Typography, Box, Container, Divider, IconButton, InputAdornment, Alert, Checkbox, FormControlLabel } from '@mui/material';
import { Visibility, VisibilityOff, Mail, Lock } from '@mui/icons-material';
import TrialSelectionModal from '../components/subscription/TrialSelectionModal';
import {
  Building,
  Star,
} from 'lucide-react';

const LoginPage: React.FC = () => {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [showPassword, setShowPassword] = useState(false);
  const [rememberMe, setRememberMe] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [trialModalOpen, setTrialModalOpen] = useState(false);

  const navigate = useNavigate();
  const isMobile = window.innerWidth < 640;

  const handleTogglePasswordVisibility = () => {
    setShowPassword(!showPassword);
  };

  const validateForm = () => {
    if (!email) {
      setError('メールアドレスを入力してください');
      return false;
    }

    if (!password) {
      setError('パスワードを入力してください');
      return false;
    }

    // 簡易的なメールアドレスの検証
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    if (!emailRegex.test(email)) {
      setError('有効なメールアドレスを入力してください');
      return false;
    }

    return true;
  };

  const handleLogin = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);

    if (!validateForm()) {
      return;
    }

    setLoading(true);

    try {
      // APIによる認証処理をここに実装
      // 例: const response = await api.post('/auth/login', { email, password });

      // 認証成功時の処理（デモ用に直接遷移）
      setTimeout(() => {
        setLoading(false);
        navigate('/dashboard');
      }, 1500);

    } catch (err) {
      setLoading(false);
      setError('ログインに失敗しました。メールアドレスとパスワードを確認してください。');
      console.error('Login error:', err);
    }
  };

  const handleOpenTrialModal = () => {
    setTrialModalOpen(true);
  };

  const handleCloseTrialModal = () => {
    setTrialModalOpen(false);
  };

  return (
    <Box
      sx={{
        minHeight: '100vh',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        backgroundColor: 'background.default',
        py: 4
      }}
    >
      <Container maxWidth="sm">
        <Paper
          elevation={3}
          sx={{
            p: 4,
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            backgroundColor: 'background.paper',
            borderRadius: 2
          }}
        >
          <Typography
            component="h1"
            variant="h4"
            gutterBottom
            sx={{ color: 'primary.main', fontWeight: 'bold', mb: 3 }}
          >
            ウェルネスアナライザー
          </Typography>

          <Typography variant="h5" gutterBottom sx={{ mb: 3 }}>
            ログイン
          </Typography>

          {error && (
            <Alert severity="error" sx={{ width: '100%', mb: 3 }}>
              {error}
            </Alert>
          )}

          <Box component="form" onSubmit={handleLogin} sx={{ width: '100%' }}>
            <TextField
              margin="normal"
              required
              fullWidth
              id="email"
              label="メールアドレス"
              name="email"
              autoComplete="email"
              autoFocus
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              InputProps={{
                startAdornment: (
                  <InputAdornment position="start">
                    <Mail />
                  </InputAdornment>
                ),
              }}
            />
            <TextField
              margin="normal"
              required
              fullWidth
              name="password"
              label="パスワード"
              type={showPassword ? 'text' : 'password'}
              id="password"
              autoComplete="current-password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              InputProps={{
                startAdornment: (
                  <InputAdornment position="start">
                    <Lock />
                  </InputAdornment>
                ),
                endAdornment: (
                  <InputAdornment position="end">
                    <IconButton
                      aria-label="toggle password visibility"
                      onClick={handleTogglePasswordVisibility}
                      edge="end"
                    >
                      {showPassword ? <VisibilityOff /> : <Visibility />}
                    </IconButton>
                  </InputAdornment>
                ),
              }}
            />

            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mt: 1 }}>
              <FormControlLabel
                control={
                  <Checkbox
                    value="remember"
                    color="primary"
                    checked={rememberMe}
                    onChange={(e) => setRememberMe(e.target.checked)}
                  />
                }
                label="ログイン状態を保存"
              />
              <Link to="/forgot-password" style={{ textDecoration: 'none' }}>
                <Typography variant="body2" color="primary">
                  パスワードをお忘れですか？
                </Typography>
              </Link>
            </Box>

            <Button
              type="submit"
              fullWidth
              variant="contained"
              size="large"
              sx={{ mt: 3, mb: 2, py: 1.5 }}
              disabled={loading}
            >
              {loading ? 'ログイン中...' : 'ログイン'}
            </Button>

            <Divider sx={{ my: 2 }}>
              <Typography variant="body2" color="text.secondary">
                または
              </Typography>
            </Divider>

            <Box sx={{ textAlign: 'center' }}>
              <Typography variant="body1" sx={{ mb: 1 }}>
                アカウントをお持ちでない方
              </Typography>
              <Button
                component={Link}
                to="/register"
                variant="outlined"
                fullWidth
                sx={{ mb: 2 }}
              >
                新規アカウント登録
              </Button>
            </Box>
          </Box>
        </Paper>

        <Box sx={{ mt: 3, textAlign: 'center' }}>
          <Typography variant="body2" color="text.secondary">
            &copy; {new Date().getFullYear()} スタートアップウェルネス分析プラットフォーム
          </Typography>
        </Box>
      </Container>

      {/* トライアル選択モーダル */}
      <TrialSelectionModal open={trialModalOpen} onClose={handleCloseTrialModal} />
    </Box>
  );
};

export default LoginPage;