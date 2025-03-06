import React, { useState } from 'react';
import {
  Dialog,
  DialogContent,
  Typography,
  Box,
  Button,
  Divider,
  Paper,
  Stepper,
  Step,
  StepLabel,
  TextField,
  Grid,
  Alert,
  Slide,
  CircularProgress,
  Chip,
} from '@mui/material';
import { TransitionProps } from '@mui/material/transitions';
import { useNavigate } from 'react-router-dom';
import CreditCardIcon from '@mui/icons-material/CreditCard';
import StarOutlineIcon from '@mui/icons-material/StarOutline';
import LockIcon from '@mui/icons-material/Lock';
import CheckCircleOutlineIcon from '@mui/icons-material/CheckCircleOutline';
import VerifiedUserIcon from '@mui/icons-material/VerifiedUser';
import CloseIcon from '@mui/icons-material/Close';
import { createUserWithEmailAndPassword } from 'firebase/auth';
import { auth } from '../../firebase/config';
import api from '../../services/api';

// スライドトランジション
const Transition = React.forwardRef(function Transition(
  props: TransitionProps & {
    children: React.ReactElement;
  },
  ref: React.Ref<unknown>,
) {
  return <Slide direction="up" ref={ref} {...props} />;
});

interface TrialSelectionModalProps {
  open: boolean;
  onClose: () => void;
}

interface TrialCardProps {
  title: string;
  description: string;
  features: string[];
  period: string;
  icon: React.ReactNode;
  requireCC: boolean;
  primaryColor: string;
  secondaryColor: string;
  onSelect: () => void;
}

const TrialCard: React.FC<TrialCardProps> = ({
  title,
  description,
  features,
  period,
  icon,
  requireCC,
  primaryColor,
  secondaryColor,
  onSelect,
}) => {
  return (
    <Paper
      elevation={3}
      sx={{
        p: 3,
        height: '100%',
        display: 'flex',
        flexDirection: 'column',
        transition: 'transform 0.2s',
        '&:hover': {
          transform: 'translateY(-5px)',
          boxShadow: 6,
        },
        borderTop: `4px solid ${primaryColor}`,
      }}
    >
      <Box
        sx={{
          display: 'flex',
          alignItems: 'center',
          mb: 2,
        }}
      >
        <Box
          sx={{
            bgcolor: secondaryColor,
            color: primaryColor,
            p: 1,
            borderRadius: 2,
            mr: 2,
          }}
        >
          {icon}
        </Box>
        <Typography variant="h6" fontWeight="bold">
          {title}
        </Typography>
      </Box>

      <Box sx={{ mb: 2 }}>
        <Chip
          label={period}
          size="small"
          sx={{ bgcolor: secondaryColor, color: 'text.primary', fontWeight: 'bold' }}
        />
      </Box>

      <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
        {description}
      </Typography>

      <Box sx={{ mb: 2, flexGrow: 1 }}>
        {features.map((feature, index) => (
          <Box key={index} sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
            <CheckCircleOutlineIcon fontSize="small" sx={{ color: primaryColor, mr: 1 }} />
            <Typography variant="body2">{feature}</Typography>
          </Box>
        ))}
      </Box>

      <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
        {requireCC ? (
          <Box sx={{ display: 'flex', alignItems: 'center' }}>
            <CreditCardIcon fontSize="small" sx={{ mr: 1 }} />
            <Typography variant="body2">クレジットカード必須</Typography>
          </Box>
        ) : (
          <Box sx={{ display: 'flex', alignItems: 'center' }}>
            <LockIcon fontSize="small" sx={{ mr: 1 }} />
            <Typography variant="body2">クレジットカード不要</Typography>
          </Box>
        )}
      </Box>

      <Button
        variant="contained"
        fullWidth
        sx={{ bgcolor: primaryColor, '&:hover': { bgcolor: primaryColor } }}
        onClick={onSelect}
      >
        このプランを選択
      </Button>
    </Paper>
  );
};

const TrialSelectionModal: React.FC<TrialSelectionModalProps> = ({ open, onClose }) => {
  const navigate = useNavigate();
  const [activeStep, setActiveStep] = useState(0);
  const [selectedTrial, setSelectedTrial] = useState<'free' | 'premium' | null>(null);
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [name, setName] = useState('');
  const [companyName, setCompanyName] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [registrationComplete, setRegistrationComplete] = useState(false);

  // 選択肢をリセット
  const handleClose = () => {
    setActiveStep(0);
    setSelectedTrial(null);
    setEmail('');
    setPassword('');
    setConfirmPassword('');
    setName('');
    setCompanyName('');
    setError('');
    setRegistrationComplete(false);
    onClose();
  };

  // トライアルタイプの選択
  const handleSelectTrial = (type: 'free' | 'premium') => {
    setSelectedTrial(type);
    setActiveStep(1);
  };

  // 登録処理
  const handleRegister = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError('');

    if (password !== confirmPassword) {
      setError('パスワードが一致しません');
      setLoading(false);
      return;
    }

    try {
      // Firebase Authで登録
      const userCredential = await createUserWithEmailAndPassword(auth, email, password);
      const user = userCredential.user;

      // バックエンドに登録情報を送信
      if (selectedTrial === 'free') {
        // 無料トライアル（クレジットカード不要）
        await api.post('/subscriptions/register/free-trial', {
          email,
          name,
          company_name: companyName,
          uid: user.uid
        });

        setRegistrationComplete(true);
        setActiveStep(2);
      } else {
        // プレミアムトライアル（クレジットカード必須）- Stripeチェックアウトページに遷移
        const response = await api.post('/subscriptions/checkout', {
          plan_id: 'premium',
          user_id: user.uid
        });

        // Stripeチェックアウトページに遷移
        window.location.href = response.data.checkout_url;
      }
    } catch (err: any) {
      console.error('登録エラー:', err);
      setError(err.message || '登録中にエラーが発生しました');
      setLoading(false);
    }
  };

  return (
    <Dialog
      open={open}
      onClose={handleClose}
      fullWidth
      maxWidth="md"
      TransitionComponent={Transition}
    >
      <Box sx={{ display: 'flex', justifyContent: 'flex-end', p: 1 }}>
        <Button startIcon={<CloseIcon />} onClick={handleClose} color="inherit">
          閉じる
        </Button>
      </Box>

      <DialogContent>
        {activeStep === 0 && (
          <>
            <Box sx={{ textAlign: 'center', mb: 4 }}>
              <Typography variant="h5" fontWeight="bold" gutterBottom>
                トライアルプランを選択
              </Typography>
              <Typography variant="body1" color="text.secondary">
                あなたのニーズに合ったトライアルプランをお選びください
              </Typography>
            </Box>

            <Grid container spacing={3}>
              <Grid item xs={12} md={6}>
                <TrialCard
                  title="スタータートライアル"
                  description="クレジットカード不要で、基本機能を7日間お試しいただけます。"
                  features={[
                    "基本的なデータ分析",
                    "最大5つのダッシュボードウィジェット",
                    "月最大5件のレポート",
                    "基本統計と相関分析",
                    "2ユーザーまで利用可能"
                  ]}
                  period="7日間"
                  icon={<StarOutlineIcon />}
                  requireCC={false}
                  primaryColor="#7986cb"
                  secondaryColor="#e8eaf6"
                  onSelect={() => handleSelectTrial('free')}
                />
              </Grid>
              <Grid item xs={12} md={6}>
                <TrialCard
                  title="プレミアムトライアル"
                  description="クレジットカードの登録が必要ですが、すべての機能を制限なく14日間お試しいただけます。"
                  features={[
                    "高度なデータ分析",
                    "無制限のダッシュボードウィジェット",
                    "無制限のレポート生成",
                    "全ての分析タイプにアクセス可能",
                    "無制限のユーザー追加",
                    "優先サポート"
                  ]}
                  period="14日間"
                  icon={<VerifiedUserIcon />}
                  requireCC={true}
                  primaryColor="#43a047"
                  secondaryColor="#e8f5e9"
                  onSelect={() => handleSelectTrial('premium')}
                />
              </Grid>
            </Grid>

            <Box sx={{ mt: 4, bgcolor: '#f5f5f5', p: 2, borderRadius: 1 }}>
              <Typography variant="body2" color="text.secondary" align="center">
                プレミアムトライアルは14日間の無料期間後、自動的に課金が開始されますが、いつでもキャンセル可能です。
              </Typography>
            </Box>
          </>
        )}

        {activeStep === 1 && (
          <>
            <Box sx={{ textAlign: 'center', mb: 4 }}>
              <Typography variant="h5" fontWeight="bold" gutterBottom>
                {selectedTrial === 'free' ? 'スタータートライアル登録' : 'プレミアムトライアル登録'}
              </Typography>
              <Typography variant="body1" color="text.secondary">
                {selectedTrial === 'free'
                  ? '基本情報を入力して、7日間の無料トライアルを開始しましょう'
                  : 'アカウント情報を入力して、14日間のフル機能トライアルを開始しましょう'}
              </Typography>
            </Box>

            {error && (
              <Alert severity="error" sx={{ mb: 3 }}>
                {error}
              </Alert>
            )}

            <Box component="form" onSubmit={handleRegister}>
              <Grid container spacing={2}>
                <Grid item xs={12} sm={6}>
                  <TextField
                    label="お名前"
                    value={name}
                    onChange={(e: React.ChangeEvent<HTMLInputElement>) => setName(e.target.value)}
                    fullWidth
                    required
                    margin="normal"
                  />
                </Grid>
                <Grid item xs={12} sm={6}>
                  <TextField
                    label="会社名"
                    value={companyName}
                    onChange={(e: React.ChangeEvent<HTMLInputElement>) => setCompanyName(e.target.value)}
                    fullWidth
                    required
                    margin="normal"
                  />
                </Grid>
                <Grid item xs={12}>
                  <TextField
                    label="メールアドレス"
                    type="email"
                    value={email}
                    onChange={(e: React.ChangeEvent<HTMLInputElement>) => setEmail(e.target.value)}
                    fullWidth
                    required
                    margin="normal"
                  />
                </Grid>
                <Grid item xs={12} sm={6}>
                  <TextField
                    label="パスワード"
                    type="password"
                    value={password}
                    onChange={(e: React.ChangeEvent<HTMLInputElement>) => setPassword(e.target.value)}
                    fullWidth
                    required
                    margin="normal"
                  />
                </Grid>
                <Grid item xs={12} sm={6}>
                  <TextField
                    label="パスワード（確認）"
                    type="password"
                    value={confirmPassword}
                    onChange={(e: React.ChangeEvent<HTMLInputElement>) => setConfirmPassword(e.target.value)}
                    fullWidth
                    required
                    margin="normal"
                  />
                </Grid>
              </Grid>

              <Box sx={{ mt: 4, mb: 2 }}>
                <Button
                  type="submit"
                  variant="contained"
                  fullWidth
                  size="large"
                  disabled={loading}
                  startIcon={selectedTrial === 'premium' ? <CreditCardIcon /> : <StarOutlineIcon />}
                >
                  {loading ? (
                    <CircularProgress size={24} />
                  ) : selectedTrial === 'free' ? (
                    '無料トライアルを開始'
                  ) : (
                    'カード情報の入力に進む'
                  )}
                </Button>
              </Box>

              <Typography variant="body2" color="text.secondary" align="center">
                {selectedTrial === 'free'
                  ? 'クレジットカード情報は必要ありません。7日間のトライアル後、自動更新はされません。'
                  : 'この後、安全なStripeの決済ページにてクレジットカード情報をご入力いただきます。14日間は無料でご利用いただけます。'}
              </Typography>
            </Box>
          </>
        )}

        {activeStep === 2 && (
          <Box sx={{ textAlign: 'center', py: 4 }}>
            <CheckCircleOutlineIcon color="success" sx={{ fontSize: 60, mb: 2 }} />
            <Typography variant="h5" fontWeight="bold" gutterBottom>
              登録完了！
            </Typography>
            <Typography variant="body1" gutterBottom>
              {selectedTrial === 'free'
                ? '7日間の無料トライアルが開始されました。'
                : '14日間のプレミアムトライアルが開始されました。'}
            </Typography>
            <Typography variant="body2" color="text.secondary" sx={{ mb: 4 }}>
              メールアドレスに確認メールを送信しました。
            </Typography>

            <Button
              variant="contained"
              color="primary"
              size="large"
              onClick={() => navigate('/dashboard')}
            >
              ダッシュボードに進む
            </Button>
          </Box>
        )}

        {activeStep > 0 && activeStep < 2 && (
          <Box sx={{ display: 'flex', justifyContent: 'flex-start', mt: 2 }}>
            <Button onClick={() => setActiveStep(activeStep - 1)}>戻る</Button>
          </Box>
        )}
      </DialogContent>
    </Dialog>
  );
};

export default TrialSelectionModal;