// Type definitions for the Lantern Fly Tracker application

export interface User {
  id: number;
  username: string;
  email?: string;
  points: number;
  total_photos: number;
  confirmed_lantern_flies: number;
  created_at: string;
  updated_at: string;
}

export interface Photo {
  id: number;
  user_id: number;
  image_url: string;
  thumbnail_url?: string;
  latitude?: number;
  longitude?: number;
  location_name?: string;
  is_lantern_fly?: boolean;
  confidence_score?: number;
  points_awarded: number;
  created_at: string;
  updated_at: string;
}

export interface PhotoUpload {
  file: File;
  latitude?: number;
  longitude?: number;
  location_name?: string;
}

export interface ClassificationResult {
  is_lantern_fly: boolean;
  confidence_score: number;
  points_awarded: number;
  model_version?: string;
  class_probabilities?: {
    non_lanternfly: number;
    lanternfly: number;
  };
}

export interface AuthResponse {
  token: string;
  user: User;
}

export interface LoginCredentials {
  username: string;
  password: string;
}

export interface RegisterCredentials {
  username: string;
  password: string;
  email?: string;
}

export interface MapLocation {
  id: number;
  latitude: number;
  longitude: number;
  is_lantern_fly: boolean;
  confidence_score: number;
  created_at: string;
}

export interface Achievement {
  id: number;
  user_id: number;
  achievement_type: string;
  achievement_name: string;
  description: string;
  points_reward: number;
  earned_at: string;
}

export interface LeaderboardEntry {
  id: number;
  username: string;
  points: number;
  total_photos: number;
  confirmed_lantern_flies: number;
  accuracy_percentage: number;
}

export interface ApiResponse<T> {
  success: boolean;
  data?: T;
  error?: string;
  message?: string;
}
