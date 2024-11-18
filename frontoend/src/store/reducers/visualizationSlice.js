import { createSlice, createAsyncThunk } from '@reduxjs/toolkit';
import VisualizationService from '../../services/VisualizationService';

export const fetchVisualizationData = createAsyncThunk(
  'visualization/fetchVisualizationData',
  async ({ startupId, visualizationType, dataPoints, timeRange }, { rejectWithValue }) => {
    try {
      const response = await VisualizationService.getVisualizationData({
        startupId,
        visualizationType,
        dataPoints,
        timeRange,
      });
      return response.data;
    } catch (error) {
      return rejectWithValue(error.response.data);
    }
  }
);


const initialState = {
  visualizationData: [],
  availableStartups: [],
  selectedStartup: null,
  availableVisualizationTypes: [],
  selectedVisualizationType: null,
  availableDataPoints: [],
  selectedDataPoints: [],
  timeRange: [2015, 2025],
  status: 'idle',
  error: null,
};

const visualizationSlice = createSlice({
  name: 'visualization',
  initialState,
  reducers: {
    setSelectedStartup: (state, action) => {
      state.selectedStartup = action.payload;
    },
    setSelectedVisualizationType: (state, action) => {
      state.selectedVisualizationType = action.payload;
    },
    setSelectedDataPoints: (state, action) => {
      state.selectedDataPoints = action.payload;
    },
    setTimeRange: (state, action) => {
      state.timeRange = action.payload;
    },
    setAvailableStartups: (state, action) => {
      state.availableStartups = action.payload;
    },
    setAvailableVisualizationTypes: (state, action) => {
      state.availableVisualizationTypes = action.payload;
    },
  },
  extraReducers(builder) {
    builder
      .addCase(fetchVisualizationData.pending, (state) => {
        state.status = 'loading';
      })
      .addCase(fetchVisualizationData.fulfilled, (state, action) => {
        state.status = 'succeeded';
        state.visualizationData = action.payload;
      })
      .addCase(fetchVisualizationData.rejected, (state, action) => {
        state.status = 'failed';
        state.error = action.payload;
      });
  },
});


export const {
  setSelectedStartup,
  setSelectedVisualizationType,
  setSelectedDataPoints,
  setTimeRange,
  setAvailableStartups,
  setAvailableVisualizationTypes,
} = visualizationSlice.actions;
export default visualizationSlice.reducer;