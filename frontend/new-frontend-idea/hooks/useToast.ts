"use client";

import { useContext } from 'react';
import { ToastContext, ToastContextType } from '../contexts/ToastContext';

/**
 * トースト通知を表示するためのカスタムフック
 *
 * @example
 * const { toast } = useToast();
 * toast({
 *   title: '成功',
 *   description: '操作が完了しました',
 *   variant: 'default'
 * });
 */
export const useToast = (): ToastContextType => {
  const context = useContext(ToastContext);

  if (context === undefined) {
    throw new Error('useToast must be used within a ToastProvider');
  }

  return context;
};