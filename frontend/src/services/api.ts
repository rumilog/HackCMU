import axios, { AxiosInstance, AxiosResponse } from 'axios';
import { ApiResponse } from '../types';

// Create axios instance
const api: AxiosInstance = axios.create({
  baseURL: import.meta.env.VITE_API_URL || 'http://localhost:5000/api',
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Request interceptor to add auth token
api.interceptors.request.use(
  (config) => {
    const token = localStorage.getItem('token');
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

// Response interceptor for error handling
api.interceptors.response.use(
  (response: AxiosResponse) => {
    return response;
  },
  (error) => {
    if (error.response?.status === 401) {
      // Token expired or invalid
      localStorage.removeItem('token');
      window.location.href = '/login';
    }
    return Promise.reject(error);
  }
);

// Generic API methods
export const apiService = {
  get: async <T>(url: string): Promise<T> => {
    const response = await api.get(url);
    return response.data;
  },

  post: async <T>(url: string, data?: any): Promise<T> => {
    const response = await api.post(url, data);
    return response.data;
  },

  put: async <T>(url: string, data?: any): Promise<T> => {
    const response = await api.put(url, data);
    return response.data;
  },

  delete: async <T>(url: string): Promise<T> => {
    const response = await api.delete(url);
    return response.data;
  },

  upload: async <T>(url: string, formData: FormData): Promise<T> => {
    // Create a separate axios instance for uploads to avoid header conflicts
    const uploadApi = axios.create({
      baseURL: import.meta.env.VITE_API_URL || 'http://localhost:5000/api',
      timeout: 30000,
    });
    
    // Add auth token if available
    const token = localStorage.getItem('token');
    if (token) {
      uploadApi.defaults.headers.Authorization = `Bearer ${token}`;
    }
    
    const response = await uploadApi.post(url, formData);
    return response.data;
  },
};

export default api;
