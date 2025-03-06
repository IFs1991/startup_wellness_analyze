import React from 'react';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { APISettings } from '@/components/settings/APISettings';
import { SecuritySettings } from '@/components/settings/SecuritySettings';
import { NotificationSettings } from '@/components/settings/NotificationSettings';
import { UserManagement } from '@/components/settings/UserManagement';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';

const SettingsPage: React.FC = () => {
  return (
    <div className="container mx-auto">
      <div className="my-8">
        <h1 className="text-3xl font-bold mb-6">
          設定
        </h1>

        <div className="grid gap-6">
          <Card>
            <CardHeader>
              <CardTitle>アカウント設定</CardTitle>
            </CardHeader>
            <CardContent>
              <p>
                アカウント設定の内容がここに表示されます。
              </p>
            </CardContent>
          </Card>
        </div>

        <Tabs defaultValue="api" className="space-y-6 mt-6">
          <TabsList>
            <TabsTrigger value="api">API設定</TabsTrigger>
            <TabsTrigger value="security">セキュリティ</TabsTrigger>
            <TabsTrigger value="notifications">通知</TabsTrigger>
            <TabsTrigger value="users">ユーザー管理</TabsTrigger>
          </TabsList>

          <TabsContent value="api">
            <APISettings />
          </TabsContent>

          <TabsContent value="security">
            <SecuritySettings />
          </TabsContent>

          <TabsContent value="notifications">
            <NotificationSettings />
          </TabsContent>

          <TabsContent value="users">
            <UserManagement />
          </TabsContent>
        </Tabs>
      </div>
    </div>
  );
};

export default SettingsPage;