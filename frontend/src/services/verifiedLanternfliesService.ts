import api from './api';

export interface VerifiedLanternflySighting {
  id: number;
  username: string;
  latitude: number;
  longitude: number;
  location_name: string | null;
  image_url: string;
  confidence_score: number;
  points_awarded: number;
  sighting_date: string;
  model_version?: string;
}

export interface VerifiedLanternfliesResponse {
  success: boolean;
  data: {
    sightings: VerifiedLanternflySighting[];
    pagination: {
      total: number;
      page: number;
      limit: number;
      totalPages: number;
    };
  };
}

export interface VerifiedLanternfliesStats {
  success: boolean;
  data: {
    totalSightings: number;
    recentSightings: number;
    usersWithSightings: number;
    sightingsByUser: Array<{
      username: string;
      count: number;
    }>;
  };
}

export const verifiedLanternfliesService = {
  /**
   * Get all verified lanternfly sightings
   */
  async getAllSightings(page: number = 1, limit: number = 50): Promise<VerifiedLanternfliesResponse> {
    const response = await api.get(`/verified-lanternflies/all?page=${page}&limit=${limit}`);
    return response.data;
  },

  /**
   * Get verified lanternfly sightings for a specific user
   */
  async getUserSightings(userId: number, page: number = 1, limit: number = 20): Promise<VerifiedLanternfliesResponse> {
    const response = await api.get(`/verified-lanternflies/user/${userId}?page=${page}&limit=${limit}`);
    return response.data;
  },

  /**
   * Get statistics for verified lanternfly sightings
   */
  async getStats(): Promise<VerifiedLanternfliesStats> {
    const response = await api.get('/verified-lanternflies/stats');
    return response.data;
  }
};
