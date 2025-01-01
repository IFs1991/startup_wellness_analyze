import { apiClient } from './client';
import type { 
  UserResponse, 
  LoginRequest, 
  RegisterRequest, 
  TokenResponse 
} from './types/auth';

export const authApi = {
  register: async (data: RegisterRequest): Promise<UserResponse> => {
    const response = await apiClient.post('/auth/register', data);
    return response.data;
  },

  login: async (data: LoginRequest): Promise<string> => {
    const response = await apiClient.post<TokenResponse>('/auth/token', data);
    const { access_token } = response.data;
    localStorage.setItem('token', access_token);
    return access_token;
  },

  logout: async () => {
    await apiClient.post('/auth/logout');
    localStorage.removeItem('token');
  },

  getProfile: async (): Promise<UserResponse> => {
    const response = await apiClient.get('/auth/me');
    return response.data;
  },

  updateProfile: async (data: Partial<UserResponse>): Promise<UserResponse> => {
    const response = await apiClient.put('/auth/me', data);
    return response.data;
  },
};