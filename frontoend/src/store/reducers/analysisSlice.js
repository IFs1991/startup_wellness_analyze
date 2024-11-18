import { createSlice, createAsyncThunk } from '@reduxjs/toolkit';
import AnalysisService from '../../services/AnalysisService';

// 非同期処理のthunkを作成 (例: 分析の実行)
export const runAnalysis = createAsyncThunk(
  'analysis/runAnalysis',
  async ({ startupId, analysisType }, { rejectWithValue }) => {
    try {
      const response = await AnalysisService.runAnalysis(startupId, analysisType);
      return response.data;
    } catch (error) {
      return rejectWithValue(error.response.data);
    }
  }
);


const analysisSlice = createSlice({
  name: 'analysis',
  initialState: {
    analysisResults: [],
    analysisSettings: {},
    status: 'idle',
    error: null,
  },

  reducers: {
    updateAnalysisSettings: (state, action) => {
      state.analysisSettings = action.payload;
    },
    // 他のreducer関数を定義
  },

  extraReducers(builder) {
    builder
      .addCase(runAnalysis.pending, (state) => {
        state.status = 'loading';
      })
      .addCase(runAnalysis.fulfilled, (state, action) => {
        state.status = 'succeeded';
        state.analysisResults = action.payload;
      })
      .addCase(runAnalysis.rejected, (state, action) => {
        state.status = 'failed';
        state.error = action.payload;
      });
  },
});

export const { updateAnalysisSettings } = analysisSlice.actions;
export default analysisSlice.reducer;