import { Card } from '@/components/ui/card';
import { Label } from '@/components/ui/label';
import { Input } from '@/components/ui/input';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Switch } from '@/components/ui/switch';
import { Button } from '@/components/ui/button';
import { Info } from 'lucide-react';
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from '@/components/ui/tooltip';

export function APISettings() {
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
            <Input id="apiKey" type="password" placeholder="sk-..." />
          </div>

          <div className="space-y-2">
            <Label htmlFor="model">モデル</Label>
            <Select defaultValue="gpt-4">
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
              <Input id="requestLimit" type="number" min="1" defaultValue="1000" />
              <Select defaultValue="hour">
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
            <Switch />
          </div>
        </div>
      </Card>
    </div>
  );
}