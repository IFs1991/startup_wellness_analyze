import axios from 'axios';

const API_URL = '/api/auth/';

const AuthService = {
  login: async (userData) => {
    try {
      const response = await axios.post(API_URL + 'token', userData);
      if (response.data.access_token) {
        localStorage.setItem('user', JSON.stringify(response.data));
      }
      return response.data;
    } catch (error) {
      throw error;
    }
  },

  logout: () => {
    localStorage.removeItem('user');
  },

  register: async (userData) => {
    try {
      const response = await axios.post(API_URL + 'register', userData);
      return response.data;
    } catch (error) {
      throw error;
    }
  },

  getCurrentUser: () => {
    const user = JSON.parse(localStorage.getItem('user'));
    return user;
  },
};

export default AuthService;