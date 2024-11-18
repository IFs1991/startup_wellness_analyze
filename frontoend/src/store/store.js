import { configureStore } from '@reduxjs/toolkit';
import analysisReducer from './reducers/analysisSlice';
import visualizationReducer from './reducers/visualizationSlice';
import reportReducer from './reducers/reportSlice';

const store = configureStore({
  reducer: {
    analysis: analysisReducer,
    visualization: visualizationReducer,
    report: reportReducer,
    // 他のreducerを追加
  },
});

export default store;