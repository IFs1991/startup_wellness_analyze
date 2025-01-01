import { Card } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Button } from '@/components/ui/button';
import { Switch } from '@/components/ui/switch';

export function SettingsPage() {
  return (
    <div className="space-y-6">
      <h1 className="text-2xl font-bold text-[#212121]">設定</h1>
      
      <Card className="p-6">
        <h2 className="text-xl font-semibold mb-4">API設定</h2>
        <div className="space-y-4">
          <div className="space-y-2">
            <Label htmlFor="apiKey">APIキー</Label>
            <Input id="apiKey" type="password" placeholder="APIキーを入力" />
          </div>
          <div className="space-y-2">
            <Label htmlFor="endpoint">エンドポイント</Label>
            <Input id="endpoint" placeholder="https://api.example.com" />
          </div>
        </div>
      </Card>
      
      <Card className="p-6">
        <h2 className="text-xl font-semibold mb-4">分析設定</h2>
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <div className="space-y-0.5">
              <Label>自動更新</Label>
              <p className="text-sm text-[#757575]">データを自動的に更新する</p>
            </div>
            <Switch />
          </div>
          <div className="space-y-2">
            <Label htmlFor="interval">更新間隔（分）</Label>
            <Input id="interval" type="number" min="1" defaultValue="5" />
          </div>
        </div>
      </Card>
      
      <div className="flex justify-end space-x-4">
        <Button variant="outline">キャンセル</Button>
        <Button>保存</Button>
      </div>
    </div>
  );
}