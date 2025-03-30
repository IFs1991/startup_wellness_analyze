import React from 'react';
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
  DialogFooter,
} from '@/components/ui/dialog';
import { Button } from '@/components/ui/button';
import { Card, CardContent } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Info, AlertTriangle, CheckCircle } from 'lucide-react';

interface ExplanationPopupProps {
  isOpen: boolean;
  onClose: () => void;
  content: {
    title: string;
    description: string;
    businessValue: string;
    caution: string;
  };
}

export function ExplanationPopup({ isOpen, onClose, content }: ExplanationPopupProps) {
  return (
    <Dialog open={isOpen} onOpenChange={onClose}>
      <DialogContent className="sm:max-w-[620px] max-h-[80vh] overflow-hidden flex flex-col">
        <DialogHeader>
          <DialogTitle className="text-xl">{content.title}</DialogTitle>
          <DialogDescription className="text-sm">
            分析手法の説明とビジネス活用方法
          </DialogDescription>
        </DialogHeader>

        <div className="flex-1 overflow-auto py-4">
          <div className="space-y-4">
            <Card>
              <CardContent className="pt-4">
                <h3 className="flex items-center gap-2 font-semibold mb-2">
                  <Info className="h-4 w-4 text-blue-500" />
                  分析手法の概要
                </h3>
                <p className="text-sm">{content.description}</p>
              </CardContent>
            </Card>

            <Card>
              <CardContent className="pt-4">
                <h3 className="flex items-center gap-2 font-semibold mb-2">
                  <CheckCircle className="h-4 w-4 text-green-500" />
                  ビジネスへの活用
                </h3>
                <p className="text-sm">{content.businessValue}</p>
              </CardContent>
            </Card>

            <Card>
              <CardContent className="pt-4">
                <h3 className="flex items-center gap-2 font-semibold mb-2">
                  <AlertTriangle className="h-4 w-4 text-amber-500" />
                  解釈時の注意点
                </h3>
                <p className="text-sm">{content.caution}</p>
              </CardContent>
            </Card>

            <div className="px-1 mt-2">
              <h4 className="text-xs font-medium text-muted-foreground mb-2">
                関連する分析手法
              </h4>
              <div className="flex flex-wrap gap-2">
                <Badge variant="outline" className="text-blue-500 bg-blue-50">統計分析</Badge>
                <Badge variant="outline" className="text-purple-500 bg-purple-50">機械学習</Badge>
                <Badge variant="outline" className="text-green-500 bg-green-50">予測モデリング</Badge>
              </div>
            </div>
          </div>
        </div>

        <DialogFooter className="pt-4 border-t">
          <Button onClick={onClose}>閉じる</Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}