import { Card } from '@/components/ui/card';
import { Label } from '@/components/ui/label';
import { Input } from '@/components/ui/input';
import { Switch } from '@/components/ui/switch';
import { Checkbox } from '@/components/ui/checkbox';

const notificationTypes = [
  {
    id: 'report',
    label: '分析レポート生成完了',
    description: '新しい分析レポートが生成された時に通知',
  },
  {
    id: 'alert',
    label: '異常検知アラート',
    description: 'システムが異常を検知した時に通知',
  },
  {
    id: 'chat',
    label: '新着メッセージ',
    description: 'チャットに新しいメッセージが届いた時に通知',
  },
  {
    id: 'mention',
    label: 'メンション',
    description: 'チャットでメンションされた時に通知',
  },
];

export function NotificationSettings() {
  return (
    <div className="space-y-6">
      <Card className="p-6">
        <h2 className="text-xl font-semibold mb-4">メール通知設定</h2>
        <div className="space-y-6">
          <div className="space-y-2">
            <Label htmlFor="email">通知先メールアドレス</Label>
            <Input id="email" type="email" placeholder="example@company.com" />
          </div>

          <div className="space-y-4">
            <Label>通知を受け取るイベント</Label>
            {notificationTypes.map((type) => (
              <div key={type.id} className="flex items-start space-x-2">
                <Checkbox id={type.id} />
                <div className="grid gap-1.5 leading-none">
                  <label
                    htmlFor={type.id}
                    className="text-sm font-medium leading-none peer-disabled:cursor-not-allowed peer-disabled:opacity-70"
                  >
                    {type.label}
                  </label>
                  <p className="text-sm text-muted-foreground">
                    {type.description}
                  </p>
                </div>
              </div>
            ))}
          </div>

          <div className="flex items-center justify-between">
            <div className="space-y-0.5">
              <Label>ダイジェストメール</Label>
              <p className="text-sm text-muted-foreground">
                1日の活動をまとめて通知
              </p>
            </div>
            <Switch />
          </div>
        </div>
      </Card>
    </div>
  );
}