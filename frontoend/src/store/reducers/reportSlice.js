import { createSlice, createAsyncThunk } from "@reduxjs/toolkit";
import ReportService from '../../services/ReportService';

export const generateReport = createAsyncThunk(
  'report/generateReport',
  async ({ startupId, reportType, additionalComments }, { rejectWithValue }) => {
    try {
      const response = await ReportService.generateReport({
        startupId,
        reportType,
        additionalComments,
      });
      return response;
    } catch (error) {
      return rejectWithValue(error.response.data);
    }
  }
);

const initialState = {
  generatedReportUrl: null,
  selectedStartup: null,
  availableStartups: [],
  selectedReportType: null,
  availableReportTypes: [],
  additionalComments: '',
  status: 'idle',
  error: null,
};

const reportSlice = createSlice({
  name: 'report',
  initialState,
  reducers: {
    setSelectedStartup: (state, action) => {
      state.selectedStartup = action.payload;
    },
    setAvailableStartups: (state, action) => {
      state.availableStartups = action.payload;
    },
    setSelectedReportType: (state, action) => {
      state.selectedReportType = action.payload;
      state.generatedReportUrl = null;
    },
    setAvailableReportTypes: (state, action) => {
      state.availableReportTypes = action.payload;
    },
    setAdditionalComments: (state, action) => {
      state.additionalComments = action.payload;
    },
  },
  extraReducers(builder) {
    builder
      .addCase(generateReport.pending, (state) => {
        state.status = 'loading';
      })
      .addCase(generateReport.fulfilled, (state, action) => {
        state.status = 'succeeded';
        state.generatedReportUrl = action.payload;
      })
      .addCase(generateReport.rejected, (state, action) => {
        state.status = 'failed';
        state.error = action.payload;
      });
  },
});


export const {
  setSelectedStartup,
  setAvailableStartups,
  setSelectedReportType,
  setAvailableReportTypes,
  setAdditionalComments,
} = reportSlice.actions;

export default reportSlice.reducer;