import { Card } from '@/components/ui/card';
import { Label } from '@/components/ui/label';
import { Input } from '@/components/ui/input';
import { Switch } from '@/components/ui/switch';
import { Textarea } from '@/components/ui/textarea';
import { Button } from '@/components/ui/button';

export function SecuritySettings() {
  return (
    <div className="space-y-6">
      <Card className="p-6">
        <h2 className="text-xl font-semibold mb-4">パスワードポリシー</h2>
        <div className="space-y-4">
          <div className="space-y-2">
            <Label htmlFor="minLength">最小文字数</Label>
            <Input id="minLength" type="number" min="8" defaultValue="12" />
          </div>

          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <Label>大文字を含める</Label>
              <Switch />
            </div>
            <div className="flex items-center justify-between">
              <Label>小文字を含める</Label>
              <Switch />
            </div>
            <div className="flex items-center justify-between">
              <Label>数字を含める</Label>
              <Switch />
            </div>
            <div className="flex items-center justify-between">
              <Label>記号を含める</Label>
              <Switch />
            </div>
          </div>

          <div className="space-y-2">
            <Label htmlFor="expireDays">パスワード有効期限（日数）</Label>
            <Input id="expireDays" type="number" min="0" defaultValue="90" />
          </div>
        </div>
      </Card>

      <Card className="p-6">
        <h2 className="text-xl font-semibold mb-4">IPアドレス制限</h2>
        <div className="space-y-4">
          <div className="flex items-center justify-between mb-4">
            <Label>IPアドレス制限を有効にする</Label>
            <Switch />
          </div>

          <div className="space-y-2">
            <Label htmlFor="allowedIps">許可するIPアドレス（1行につき1つ）</Label>
            <Textarea
              id="allowedIps"
              placeholder="例:&#10;192.168.1.1&#10;10.0.0.0/24"
              className="h-32"
            />
          </div>
        </div>
      </Card>
    </div>
  );
}