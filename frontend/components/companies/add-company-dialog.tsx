import React from "react";
import { Dialog, DialogTitle, DialogContent } from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Select, SelectItem, SelectTrigger, SelectContent } from "@/components/ui/select";
import { useForm } from "react-hook-form";
import { Form, FormField, FormItem, FormLabel, FormControl, FormMessage } from "@/components/ui/form";

interface AddCompanyDialogProps {
  open: boolean;
  onClose: () => void;
  onSubmit: (company: CompanyFormData) => void;
}

export interface CompanyFormData {
  name: string;
  ceo: string;
  foundedYear: string;
  funding: string;
  stage: string;
  industry: string;
  sector: string;
}

const STAGE_OPTIONS = [
  "シード",
  "アーリー",
  "ミドル",
  "レイター",
  "IPO済み",
];

export const AddCompanyDialog: React.FC<AddCompanyDialogProps> = ({ open, onClose, onSubmit }) => {
  const form = useForm<CompanyFormData>({
    defaultValues: {
      name: "",
      ceo: "",
      foundedYear: "",
      funding: "",
      stage: STAGE_OPTIONS[0],
      industry: "",
      sector: "",
    },
    mode: "onBlur",
  });

  const handleSubmit = (data: CompanyFormData) => {
    onSubmit(data);
    form.reset();
    onClose();
  };

  return (
    <Dialog open={open} onOpenChange={onClose}>
      <DialogTitle>企業を追加</DialogTitle>
      <Form {...form}>
        <form onSubmit={form.handleSubmit(handleSubmit)}>
          <DialogContent className="space-y-4">
            <FormField
              control={form.control}
              name="name"
              rules={{ required: "社名は必須です" }}
              render={({ field }) => (
                <FormItem>
                  <FormLabel htmlFor="name">社名</FormLabel>
                  <FormControl>
                    <Input id="name" placeholder="例：株式会社サンプル" {...field} />
                  </FormControl>
                  <FormMessage />
                </FormItem>
              )}
            />
            <FormField
              control={form.control}
              name="ceo"
              rules={{ required: "社長名は必須です" }}
              render={({ field }) => (
                <FormItem>
                  <FormLabel htmlFor="ceo">社長名</FormLabel>
                  <FormControl>
                    <Input id="ceo" placeholder="例：山田 太郎" {...field} />
                  </FormControl>
                  <FormMessage />
                </FormItem>
              )}
            />
            <FormField
              control={form.control}
              name="foundedYear"
              rules={{ required: "設立年度は必須です", pattern: { value: /^\d{4}$/, message: "4桁の西暦で入力してください" } }}
              render={({ field }) => (
                <FormItem>
                  <FormLabel htmlFor="foundedYear">設立年度</FormLabel>
                  <FormControl>
                    <Input id="foundedYear" placeholder="例：2020" {...field} />
                  </FormControl>
                  <FormMessage />
                </FormItem>
              )}
            />
            <FormField
              control={form.control}
              name="funding"
              rules={{ required: "調達金額は必須です" }}
              render={({ field }) => (
                <FormItem>
                  <FormLabel htmlFor="funding">調達金額</FormLabel>
                  <FormControl>
                    <Input id="funding" placeholder="例：1億円" {...field} />
                  </FormControl>
                  <FormMessage />
                </FormItem>
              )}
            />
            <FormField
              control={form.control}
              name="stage"
              rules={{ required: "ステージは必須です" }}
              render={({ field }) => (
                <FormItem>
                  <FormLabel htmlFor="stage">ステージ</FormLabel>
                  <FormControl>
                    <Select value={field.value} onValueChange={field.onChange}>
                      <SelectTrigger className="w-full" />
                      <SelectContent>
                        {STAGE_OPTIONS.map((option) => (
                          <SelectItem key={option} value={option}>{option}</SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  </FormControl>
                  <FormMessage />
                </FormItem>
              )}
            />
            <FormField
              control={form.control}
              name="industry"
              rules={{ required: "業界は必須です" }}
              render={({ field }) => (
                <FormItem>
                  <FormLabel htmlFor="industry">業界</FormLabel>
                  <FormControl>
                    <Input id="industry" placeholder="例：IT" {...field} />
                  </FormControl>
                  <FormMessage />
                </FormItem>
              )}
            />
            <FormField
              control={form.control}
              name="sector"
              rules={{ required: "業種は必須です" }}
              render={({ field }) => (
                <FormItem>
                  <FormLabel htmlFor="sector">業種</FormLabel>
                  <FormControl>
                    <Input id="sector" placeholder="例：ソフトウェア" {...field} />
                  </FormControl>
                  <FormMessage />
                </FormItem>
              )}
            />
          </DialogContent>
          <div className="flex justify-end gap-2 p-4">
            <Button type="button" variant="ghost" onClick={onClose}>キャンセル</Button>
            <Button type="submit">登録</Button>
          </div>
        </form>
      </Form>
    </Dialog>
  );
};