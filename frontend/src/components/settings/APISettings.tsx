import { Card } from '@/components/ui/card';
import { Label } from '@/components/ui/label';
import { Input } from '@/components/ui/input';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Switch } from '@/components/ui/switch';
import { Button } from '@/components/ui/button';
import { Info } from 'lucide-react';
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from '@/components/ui/tooltip';
import { useEffect, useState } from 'react';
import { useToast } from '@/hooks/use-toast';

interface OpenAISettings {
  model: string;
  request_limit: number;
  limit_period: string;
  notify_on_limit: boolean;
}

export function APISettings() {
  const { toast } = useToast();
  const [apiKey, setApiKey] = useState('');
  const [settings, setSettings] = useState<OpenAISettings>({
    model: 'gpt-4',
    request_limit: 1000,
    limit_period: 'hour',
    notify_on_limit: true,
  });
  const [isSaving, setIsSaving] = useState(false);

  useEffect(() => {
    // 設定を取得
    fetchSettings();
  }, []);

  const fetchSettings = async () => {
    try {
      const response = await fetch('/api/v1/settings/openai');
      if (response.ok) {
        const data = await response.json();
        setSettings(data);
      }
    } catch (error) {
      console.error('設定の取得に失敗しました:', error);
    }
  };

  const handleSave = async () => {
    setIsSaving(true);
    try {
      const response = await fetch('/api/v1/settings/openai', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          api_key: apiKey,
          ...settings,
        }),
      });

      if (response.ok) {
        toast({
          title: '設定を保存しました',
          description: 'API設定が正常に更新されました。',
        });
        setApiKey(''); // セキュリティのため入力をクリア
      } else {
        throw new Error('設定の保存に失敗しました');
      }
    } catch (error) {
      toast({
        title: 'エラー',
        description: '設定の保存中にエラーが発生しました。',
        variant: 'destructive',
      });
    } finally {
      setIsSaving(false);
    }
  };

  return (
    <div className="space-y-6">
      <Card className="p-6">
        <h2 className="text-xl font-semibold mb-4">生成AI API設定</h2>
        <div className="space-y-4">
          <div className="space-y-2">
            <div className="flex items-center space-x-2">
              <Label htmlFor="apiKey">APIキー</Label>
              <TooltipProvider>
                <Tooltip>
                  <TooltipTrigger>
                    <Info className="h-4 w-4 text-muted-foreground" />
                  </TooltipTrigger>
                  <TooltipContent>
                    <p>OpenAI APIキーを入力してください</p>
                  </TooltipContent>
                </Tooltip>
              </TooltipProvider>
            </div>
            <Input
              id="apiKey"
              type="password"
              placeholder="sk-..."
              value={apiKey}
              onChange={(e) => setApiKey(e.target.value)}
            />
          </div>

          <div className="space-y-2">
            <Label htmlFor="model">モデル</Label>
            <Select
              value={settings.model}
              onValueChange={(value) => setSettings({ ...settings, model: value })}
            >
              <SelectTrigger>
                <SelectValue placeholder="モデルを選択" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="gpt-4">GPT-4</SelectItem>
                <SelectItem value="gpt-3.5-turbo">GPT-3.5 Turbo</SelectItem>
              </SelectContent>
            </Select>
          </div>
        </div>
      </Card>

      <Card className="p-6">
        <h2 className="text-xl font-semibold mb-4">利用制限設定</h2>
        <div className="space-y-4">
          <div className="space-y-2">
            <Label htmlFor="requestLimit">リクエスト数上限</Label>
            <div className="flex space-x-2">
              <Input
                id="requestLimit"
                type="number"
                min="1"
                value={settings.request_limit}
                onChange={(e) =>
                  setSettings({
                    ...settings,
                    request_limit: parseInt(e.target.value) || 1000,
                  })
                }
              />
              <Select
                value={settings.limit_period}
                onValueChange={(value) =>
                  setSettings({ ...settings, limit_period: value })
                }
              >
                <SelectTrigger className="w-[120px]">
                  <SelectValue placeholder="期間" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="hour">1時間あたり</SelectItem>
                  <SelectItem value="day">1日あたり</SelectItem>
                  <SelectItem value="month">1月あたり</SelectItem>
                </SelectContent>
              </Select>
            </div>
          </div>

          <div className="flex items-center justify-between">
            <div className="space-y-0.5">
              <Label>使用量通知</Label>
              <p className="text-sm text-muted-foreground">
                制限の80%に達した時に通知する
              </p>
            </div>
            <Switch
              checked={settings.notify_on_limit}
              onCheckedChange={(checked) =>
                setSettings({ ...settings, notify_on_limit: checked })
              }
            />
          </div>

          <Button
            className="w-full"
            onClick={handleSave}
            disabled={isSaving || !apiKey}
          >
            {isSaving ? '保存中...' : '設定を保存'}
          </Button>
        </div>
      </Card>
    </div>
  );
}