"use client";

import React, { useState, useRef } from 'react';
import { Upload, X, File, Image, FileArchive, FileText } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { cn } from '@/lib/utils';
import { Progress } from '@/components/ui/progress';

export interface FileAttachment {
  id: string;
  file: File;
  previewUrl?: string;
  uploadProgress?: number;
  status?: 'uploading' | 'uploaded' | 'error';
}

interface FileAttachmentProps {
  onFileSelect: (files: FileAttachment[]) => void;
  onFileRemove: (fileId: string) => void;
  attachments: FileAttachment[];
  multiple?: boolean;
  maxSize?: number; // MB単位
  acceptedTypes?: string[];
  className?: string;
  disabled?: boolean;
}

export function FileAttachment({
  onFileSelect,
  onFileRemove,
  attachments = [],
  multiple = true,
  maxSize = 5, // デフォルト5MB
  acceptedTypes = ['image/*', 'application/pdf', '.doc', '.docx', '.xls', '.xlsx', '.csv'],
  className,
  disabled = false
}: FileAttachmentProps) {
  const [isDragging, setIsDragging] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (!files || files.length === 0) return;

    processFiles(Array.from(files));

    // ファイル選択後にinput要素をリセット（同じファイルを連続で選択できるように）
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const validateFile = (file: File): string | null => {
    // ファイルサイズのチェック
    if (file.size > maxSize * 1024 * 1024) {
      return `ファイルサイズが大きすぎます（最大${maxSize}MB）`;
    }

    // ファイルタイプのチェック
    if (acceptedTypes.length > 0) {
      const fileType = file.type;
      const fileExtension = `.${file.name.split('.').pop()?.toLowerCase()}`;

      const isTypeAccepted = acceptedTypes.some(type => {
        if (type.startsWith('.')) {
          // 拡張子でのチェック
          return fileExtension === type.toLowerCase();
        } else if (type.includes('*')) {
          // ワイルドカードでのチェック (例: image/*)
          return new RegExp(type.replace('*', '.*')).test(fileType);
        } else {
          // 完全一致チェック
          return fileType === type;
        }
      });

      if (!isTypeAccepted) {
        return '対応していないファイル形式です';
      }
    }

    return null;
  };

  const processFiles = (fileList: File[]) => {
    const newFiles: FileAttachment[] = [];
    setError(null);

    for (const file of fileList) {
      const validationError = validateFile(file);
      if (validationError) {
        setError(validationError);
        continue;
      }

      // 画像の場合はプレビューURLを生成
      let previewUrl: string | undefined;
      if (file.type.startsWith('image/')) {
        previewUrl = URL.createObjectURL(file);
      }

      newFiles.push({
        id: `${file.name}-${Date.now()}-${Math.random().toString(36).substring(2, 9)}`,
        file,
        previewUrl,
        status: 'uploaded',
        uploadProgress: 100
      });
    }

    if (newFiles.length > 0) {
      onFileSelect(newFiles);
    }
  };

  const handleDragOver = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = () => {
    setIsDragging(false);
  };

  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragging(false);

    if (disabled) return;

    const files = e.dataTransfer.files;
    if (files.length === 0) return;

    processFiles(Array.from(files));
  };

  const handleBrowseClick = () => {
    if (fileInputRef.current) {
      fileInputRef.current.click();
    }
  };

  // ファイルタイプに基づいてアイコンを選択
  const getFileIcon = (file: File) => {
    if (file.type.startsWith('image/')) {
      return <Image className="h-4 w-4" />;
    } else if (file.type === 'application/pdf') {
      return <FileText className="h-4 w-4" />;
    } else if (
      file.type.includes('zip') ||
      file.type.includes('compressed') ||
      file.type.includes('archive')
    ) {
      return <FileArchive className="h-4 w-4" />;
    } else {
      return <File className="h-4 w-4" />;
    }
  };

  // ファイルサイズを読みやすい形式に変換
  const formatFileSize = (bytes: number): string => {
    if (bytes < 1024) return bytes + ' B';
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
    return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
  };

  return (
    <div className={cn('w-full', className)}>
      {/* ドラッグ&ドロップエリア */}
      {(!multiple || attachments.length === 0) && (
        <div
          className={cn(
            'border-2 border-dashed rounded-md p-4 transition-colors',
            isDragging ? 'border-primary bg-primary/5' : 'border-muted-foreground/20',
            disabled ? 'opacity-50 cursor-not-allowed' : 'cursor-pointer',
            'text-center'
          )}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
          onClick={disabled ? undefined : handleBrowseClick}
        >
          <Upload className="h-6 w-6 mx-auto text-muted-foreground" />
          <p className="text-sm mt-2 text-muted-foreground">
            ファイルをドラッグ&ドロップするか、クリックして選択
          </p>
          <input
            type="file"
            ref={fileInputRef}
            className="hidden"
            onChange={handleFileChange}
            multiple={multiple}
            accept={acceptedTypes.join(',')}
            disabled={disabled}
          />
        </div>
      )}

      {/* エラーメッセージ */}
      {error && (
        <p className="text-xs text-destructive mt-2">{error}</p>
      )}

      {/* 選択されたファイルのプレビュー */}
      {attachments.length > 0 && (
        <div className="mt-2 space-y-2">
          {attachments.map((attachment) => (
            <div
              key={attachment.id}
              className="flex items-center bg-muted/30 rounded-md p-2 text-sm"
            >
              {/* プレビュー画像（画像ファイルの場合） */}
              {attachment.previewUrl ? (
                <div className="h-10 w-10 bg-background rounded overflow-hidden mr-3 flex-shrink-0">
                  <img
                    src={attachment.previewUrl}
                    alt="Preview"
                    className="h-full w-full object-cover"
                  />
                </div>
              ) : (
                <div className="h-10 w-10 bg-background rounded flex items-center justify-center mr-3 flex-shrink-0">
                  {getFileIcon(attachment.file)}
                </div>
              )}

              {/* ファイル情報 */}
              <div className="flex-1 min-w-0">
                <p className="truncate font-medium">{attachment.file.name}</p>
                <p className="text-xs text-muted-foreground">
                  {formatFileSize(attachment.file.size)}
                </p>

                {/* アップロード進捗バー */}
                {attachment.status === 'uploading' && attachment.uploadProgress !== undefined && (
                  <Progress value={attachment.uploadProgress} className="h-1 mt-1" />
                )}
              </div>

              {/* 削除ボタン */}
              <Button
                variant="ghost"
                size="icon"
                className="h-6 w-6"
                onClick={() => onFileRemove(attachment.id)}
                disabled={disabled}
              >
                <X className="h-3 w-3" />
                <span className="sr-only">削除</span>
              </Button>
            </div>
          ))}
        </div>
      )}

      {/* 複数ファイルが許可されていて、すでにファイルが選択されている場合の追加ボタン */}
      {multiple && attachments.length > 0 && (
        <Button
          variant="outline"
          size="sm"
          className="mt-2"
          onClick={handleBrowseClick}
          disabled={disabled}
        >
          <Upload className="h-3 w-3 mr-1" />
          ファイルを追加
        </Button>
      )}
    </div>
  );
}