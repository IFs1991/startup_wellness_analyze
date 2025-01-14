import { useState, useEffect } from 'react';
import { Card } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Textarea } from '@/components/ui/textarea';
import { useToast } from '@/hooks/use-toast';
import { formatDate } from '@/lib/utils';

interface Note {
  id: string;
  content: string;
  createdAt: string;
}

interface CompanyNotesProps {
  companyId: string;
}

export function CompanyNotes({ companyId }: CompanyNotesProps) {
  const [notes, setNotes] = useState<Note[]>([]);
  const [newNote, setNewNote] = useState('');
  const { toast } = useToast();

  useEffect(() => {
    // TODO: Fetch notes from API
    const mockNotes = [
      { id: '1', content: '初回面談実施。製品デモを確認', createdAt: '2024-03-20T10:00:00Z' },
      { id: '2', content: '決算書の分析完了', createdAt: '2024-03-19T15:30:00Z' },
    ];
    setNotes(mockNotes);
  }, [companyId]);

  const handleAddNote = async () => {
    if (!newNote.trim()) return;

    try {
      // TODO: Implement API call
      const note = {
        id: Date.now().toString(),
        content: newNote,
        createdAt: new Date().toISOString(),
      };
      
      setNotes(prev => [note, ...prev]);
      setNewNote('');
      
      toast({
        title: 'メモを追加しました',
      });
    } catch (error) {
      toast({
        title: 'エラー',
        description: 'メモの追加に失敗しました。',
        variant: 'destructive',
      });
    }
  };

  return (
    <Card className="p-6">
      <h3 className="text-lg font-semibold mb-4">メモ</h3>
      
      <div className="space-y-4">
        <div className="space-y-2">
          <Textarea
            placeholder="新しいメモを入力..."
            value={newNote}
            onChange={(e) => setNewNote(e.target.value)}
            className="min-h-[100px]"
          />
          <Button onClick={handleAddNote} className="w-full">
            メモを追加
          </Button>
        </div>

        <div className="space-y-4">
          {notes.map((note) => (
            <div key={note.id} className="p-4 bg-muted rounded-lg">
              <p className="whitespace-pre-wrap">{note.content}</p>
              <p className="text-sm text-muted-foreground mt-2">
                {formatDate(new Date(note.createdAt))}
              </p>
            </div>
          ))}
        </div>
      </div>
    </Card>
  );
}