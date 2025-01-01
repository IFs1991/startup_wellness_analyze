import { createApi, fetchBaseQuery } from '@reduxjs/toolkit/query/react';

const baseUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

export const apiSlice = createApi({
  reducerPath: 'api',
  baseQuery: fetchBaseQuery({
    baseUrl,
    credentials: 'include',
    prepareHeaders: (headers) => {
      headers.set('Content-Type', 'application/json');
      return headers;
    },
  }),
  endpoints: (builder) => ({
    // ここにエンドポイントを追加していきます
    getWellnessData: builder.query({
      query: () => '/api/wellness',
    }),
    // 他のエンドポイントも必要に応じて追加
  }),
});

export const {
  useGetWellnessDataQuery,
  // 他のフックもエンドポイントに応じて追加
} = apiSlice;