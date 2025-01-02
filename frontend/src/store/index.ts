import { configureStore } from '@reduxjs/toolkit';

export const store = configureStore({
  reducer: {
    // ここに各スライスのリデューサーを追加
  },
});

export type RootState = ReturnType<typeof store.getState>;
export type AppDispatch = typeof store.dispatch;