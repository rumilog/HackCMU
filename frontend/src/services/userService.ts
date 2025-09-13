import { apiService } from './api';
import { User, LeaderboardEntry, Achievement } from '../types';

export const userService = {
  async getCurrentUser(): Promise<User> {
    try {
      const response = await apiService.get<User>('/users/me');
      return response;
    } catch (error: any) {
      throw new Error(error.response?.data?.message || 'Failed to fetch user data');
    }
  },

  async updateProfile(data: Partial<User>): Promise<User> {
    try {
      const response = await apiService.put<User>('/users/profile', data);
      return response;
    } catch (error: any) {
      throw new Error(error.response?.data?.message || 'Failed to update profile');
    }
  },

  async getLeaderboard(): Promise<LeaderboardEntry[]> {
    try {
      const response = await apiService.get<LeaderboardEntry[]>('/users/leaderboard');
      return response;
    } catch (error: any) {
      throw new Error(error.response?.data?.message || 'Failed to fetch leaderboard');
    }
  },

  async getUserAchievements(): Promise<Achievement[]> {
    try {
      const response = await apiService.get<Achievement[]>('/users/achievements');
      return response;
    } catch (error: any) {
      throw new Error(error.response?.data?.message || 'Failed to fetch achievements');
    }
  },

  async getUserStats(): Promise<{
    total_photos: number;
    confirmed_lantern_flies: number;
    points: number;
    accuracy_percentage: number;
  }> {
    try {
      const response = await apiService.get('/users/stats');
      return response;
    } catch (error: any) {
      throw new Error(error.response?.data?.message || 'Failed to fetch user stats');
    }
  },
};
