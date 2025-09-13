import { apiService } from './api';
import { Photo, PhotoUpload, ClassificationResult, MapLocation } from '../types';

export const photoService = {
  async uploadPhoto(photoData: PhotoUpload): Promise<Photo> {
    try {
      const formData = new FormData();
      formData.append('photo', photoData.file);
      
      if (photoData.latitude !== undefined) {
        formData.append('latitude', photoData.latitude.toString());
      }
      if (photoData.longitude !== undefined) {
        formData.append('longitude', photoData.longitude.toString());
      }
      if (photoData.location_name) {
        formData.append('location_name', photoData.location_name);
      }

      const response = await apiService.upload<Photo>('/photos/upload', formData);
      return response;
    } catch (error: any) {
      throw new Error(error.response?.data?.message || 'Photo upload failed');
    }
  },

  async classifyPhoto(file: File): Promise<ClassificationResult> {
    try {
      const formData = new FormData();
      formData.append('image', file);

      const response = await apiService.upload<ClassificationResult>('/photos/classify', formData);
      return response;
    } catch (error: any) {
      throw new Error(error.response?.data?.message || 'Photo classification failed');
    }
  },

  async getUserPhotos(page: number = 1, limit: number = 20): Promise<{ photos: Photo[]; total: number }> {
    try {
      const response = await apiService.get<{ photos: Photo[]; total: number }>(
        `/photos/user?page=${page}&limit=${limit}`
      );
      return response;
    } catch (error: any) {
      throw new Error(error.response?.data?.message || 'Failed to fetch photos');
    }
  },

  async getPhotoById(id: number): Promise<Photo> {
    try {
      const response = await apiService.get<Photo>(`/photos/${id}`);
      return response;
    } catch (error: any) {
      throw new Error(error.response?.data?.message || 'Failed to fetch photo');
    }
  },

  async deletePhoto(id: number): Promise<void> {
    try {
      await apiService.delete(`/photos/${id}`);
    } catch (error: any) {
      throw new Error(error.response?.data?.message || 'Failed to delete photo');
    }
  },

  async getMapLocations(): Promise<MapLocation[]> {
    try {
      const response = await apiService.get<MapLocation[]>('/photos/map');
      return response;
    } catch (error: any) {
      throw new Error(error.response?.data?.message || 'Failed to fetch map locations');
    }
  },
};
