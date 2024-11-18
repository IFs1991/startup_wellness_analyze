import React, { createContext, useState, useEffect } from 'react';
import DataService from '../services/DataService';

const DataContext = createContext();

const DataContextProvider = ({ children }) => {
  const [startupData, setStartupData] = useState([]);

  useEffect(() => {
    const fetchStartupData = async () => {
      try {
        const data = await DataService.getStartupData();
        setStartupData(data);
      } catch (error) {
        console.error('Failed to fetch startup data:', error);
      }
    };

    fetchStartupData();
  }, []);

  return (
    <DataContext.Provider value={{ startupData, setStartupData }}>
      {children}
    </DataContext.Provider>
  );
};

export default DataContextProvider;
export { DataContext };