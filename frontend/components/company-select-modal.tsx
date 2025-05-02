import React from "react";
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogFooter } from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";

export type Company = {
  id: string;
  name: string;
};

type Props = {
  open: boolean;
  companies: Company[];
  onSelect: (company: Company) => void;
  onClose: () => void;
};

export default function CompanySelectModal({ open, companies, onSelect, onClose }: Props) {
  return (
    <Dialog open={open} onOpenChange={onClose}>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>企業を選択</DialogTitle>
        </DialogHeader>
        <div className="space-y-2 max-h-60 overflow-y-auto">
          {companies.map(company => (
            <Button key={company.id} variant="outline" className="w-full justify-start" onClick={() => onSelect(company)}>
              {company.name}
            </Button>
          ))}
        </div>
        <DialogFooter>
          <Button variant="ghost" onClick={onClose}>キャンセル</Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}