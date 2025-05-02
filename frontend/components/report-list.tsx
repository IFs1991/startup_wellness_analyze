import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Star, Eye, Download, Edit, Trash2 } from "lucide-react";
import { useState } from "react";
import CompanySelectModal, { Company } from "./company-select-modal";
import ReportChatCreator, { ChatReportDraft } from "./report-chat-creator";

export type Report = {
  id: string;
  title: string;
  company: string;
  createdAt: string;
  author: string;
  type: string;
  favorite: boolean;
};

const dummyReports: Report[] = [
  {
    id: "1",
    title: "2024年5月 月次レポート",
    company: "テックスタート社",
    createdAt: "2024-05-31 10:00",
    author: "山田太郎",
    type: "VC向け",
    favorite: true
  },
  {
    id: "2",
    title: "2024年5月 経営分析レポート",
    company: "フューチャーラボ",
    createdAt: "2024-05-30 15:20",
    author: "佐藤花子",
    type: "経営陣向け",
    favorite: false
  },
  {
    id: "3",
    title: "2024年4月 月次レポート",
    company: "イノベーションテック",
    createdAt: "2024-04-30 09:10",
    author: "山田太郎",
    type: "VC向け",
    favorite: false
  }
];

export default function ReportList() {
  const [reports, setReports] = useState<Report[]>(dummyReports);
  const [filter, setFilter] = useState({ company: "", type: "", favorite: false });
  const [search, setSearch] = useState("");
  const dummyCompanies: Company[] = [
    { id: "1", name: "テックスタート社" },
    { id: "2", name: "フューチャーラボ" },
    { id: "3", name: "イノベーションテック" }
  ];
  const [companyModalOpen, setCompanyModalOpen] = useState(false);
  const [chatCreatorOpen, setChatCreatorOpen] = useState(false);
  const [selectedCompany, setSelectedCompany] = useState<Company | null>(null);
  const filteredReports = reports.filter(r => {
    return (
      (filter.company === "" || r.company.includes(filter.company)) &&
      (filter.type === "" || r.type === filter.type) &&
      (!filter.favorite || r.favorite) &&
      (search === "" || r.title.includes(search) || r.company.includes(search))
    );
  });

  const toggleFavorite = (id: string) => {
    setReports(reports => reports.map(r => r.id === id ? { ...r, favorite: !r.favorite } : r));
  };

  return (
    <div className="space-y-6">
      {/* 新規レポートボタン */}
      <div className="flex justify-end">
        <Button
          className="bg-blue-500 text-white"
          onClick={handleNewReport}
        >
          ＋ 新規レポート
        </Button>
      </div>
      {/* フィルタ・検索UI */}
      <div className="flex flex-wrap gap-2 items-end">
        <div>
          <label className="block text-xs font-medium">ポートフォリオ企業名</label>
          <input type="text" className="input input-bordered input-sm w-40" value={filter.company} onChange={e => setFilter(f => ({ ...f, company: e.target.value }))} />
        </div>
        <div>
          <label className="block text-xs font-medium">分析タイプ</label>
          <select className="select select-bordered select-sm w-32" value={filter.type} onChange={e => setFilter(f => ({ ...f, type: e.target.value }))}>
            <option value="">全て</option>
            <option value="VC向け">VC向け</option>
            <option value="経営陣向け">経営陣向け</option>
          </select>
        </div>
        <div>
          <label className="block text-xs font-medium">お気に入り</label>
          <input type="checkbox" className="checkbox checkbox-sm" checked={filter.favorite} onChange={e => setFilter(f => ({ ...f, favorite: e.target.checked }))} />
        </div>
        <div>
          <label className="block text-xs font-medium">検索</label>
          <input type="text" className="input input-bordered input-sm w-40" placeholder="タイトル・企業名で検索" value={search} onChange={e => setSearch(e.target.value)} />
        </div>
      </div>
      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
        {filteredReports.map(report => (
          <Card key={report.id} className="relative group">
            <CardHeader className="pb-2">
              <CardTitle className="flex items-center gap-2 text-base">
                {report.title}
                <button onClick={() => toggleFavorite(report.id)} className="ml-1 text-yellow-400 hover:text-yellow-500">
                  <Star fill={report.favorite ? "#facc15" : "none"} className="h-4 w-4" />
                </button>
              </CardTitle>
              <CardDescription className="flex flex-wrap gap-2 text-xs mt-1">
                <span>企業: {report.company}</span>
                <span>作成日: {report.createdAt}</span>
                <span>作成者: {report.author}</span>
                <span>タイプ: {report.type}</span>
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="flex gap-2 mt-2">
                <Button size="sm" variant="outline" className="flex items-center gap-1">
                  <Eye className="h-4 w-4" />プレビュー
                </Button>
                <Button size="sm" variant="outline" className="flex items-center gap-1">
                  <Download className="h-4 w-4" />ダウンロード
                </Button>
                <Button size="sm" variant="outline" className="flex items-center gap-1">
                  <Edit className="h-4 w-4" />編集
                </Button>
                <Button size="sm" variant="destructive" className="flex items-center gap-1">
                  <Trash2 className="h-4 w-4" />削除
                </Button>
              </div>
            </CardContent>
          </Card>
        ))}
      </div>
      <CompanySelectModal
        open={companyModalOpen}
        companies={dummyCompanies}
        onSelect={handleCompanySelect}
        onClose={() => setCompanyModalOpen(false)}
      />
      <ReportChatCreator
        open={chatCreatorOpen}
        company={selectedCompany?.name || ""}
        onComplete={handleChatComplete}
        onClose={() => setChatCreatorOpen(false)}
      />
    </div>
  );
}

const handleNewReport = () => {
  setCompanyModalOpen(true);
};

const handleCompanySelect = (company: Company) => {
  setSelectedCompany(company);
  setCompanyModalOpen(false);
  setTimeout(() => setChatCreatorOpen(true), 200);
};

const handleChatComplete = (draft: ChatReportDraft) => {
  setReports(reports => [
    {
      id: (reports.length + 1).toString(),
      title: draft.messages[draft.messages.length - 1]?.content?.slice(0, 20) || "新規レポート",
      company: draft.company,
      createdAt: new Date().toLocaleString("ja-JP", { hour12: false }),
      author: "AIユーザー",
      type: "VC向け",
      favorite: false
    },
    ...reports
  ]);
  setChatCreatorOpen(false);
  setSelectedCompany(null);
};