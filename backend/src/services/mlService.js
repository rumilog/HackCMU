/**
 * Service for communicating with the ML classification API
 */

import axios from 'axios';
import FormData from 'form-data';
import fs from 'fs';
import path from 'path';

class MLService {
  constructor() {
    // ML API endpoint - can be configured via environment variable
    this.mlApiUrl = process.env.ML_API_URL || 'http://localhost:5001';
    this.timeout = 30000; // 30 seconds timeout
  }

  /**
   * Check if the ML service is available
   */
  async isHealthy() {
    try {
      const response = await axios.get(`${this.mlApiUrl}/health`, {
        timeout: 5000
      });
      return response.data.status === 'healthy';
    } catch (error) {
      console.error('ML service health check failed:', error.message);
      return false;
    }
  }

  /**
   * Classify an image file using the ML model
   * @param {string} imagePath - Path to the image file
   * @returns {Promise<Object>} Classification result
   */
  async classifyImage(imagePath) {
    try {
      // Check if file exists
      if (!fs.existsSync(imagePath)) {
        throw new Error(`Image file not found: ${imagePath}`);
      }

      // Create form data
      const formData = new FormData();
      formData.append('image', fs.createReadStream(imagePath));

      // Make request to ML API
      const response = await axios.post(`${this.mlApiUrl}/classify`, formData, {
        headers: {
          ...formData.getHeaders(),
        },
        timeout: this.timeout,
        maxContentLength: 50 * 1024 * 1024, // 50MB max
      });

      if (!response.data.success) {
        throw new Error(response.data.error || 'Classification failed');
      }

      return {
        success: true,
        isLanternFly: response.data.is_lantern_fly,
        confidenceScore: response.data.confidence_score,
        pointsAwarded: response.data.points_awarded,
        modelVersion: response.data.model_version,
        classProbabilities: response.data.class_probabilities
      };

    } catch (error) {
      console.error('ML classification error:', error.message);
      
      // Return fallback result if ML service is unavailable
      if (error.code === 'ECONNREFUSED' || error.code === 'ETIMEDOUT') {
        console.warn('ML service unavailable, using fallback classification');
        return this.getFallbackResult();
      }

      throw new Error(`Classification failed: ${error.message}`);
    }
  }

  /**
   * Classify an image from buffer data
   * @param {Buffer} imageBuffer - Image data as buffer
   * @param {string} filename - Original filename
   * @returns {Promise<Object>} Classification result
   */
  async classifyImageBuffer(imageBuffer, filename = 'image.jpg') {
    try {
      // Create form data
      const formData = new FormData();
      formData.append('image', imageBuffer, {
        filename: filename,
        contentType: 'image/jpeg'
      });

      // Make request to ML API
      const response = await axios.post(`${this.mlApiUrl}/classify`, formData, {
        headers: {
          ...formData.getHeaders(),
        },
        timeout: this.timeout,
        maxContentLength: 50 * 1024 * 1024, // 50MB max
      });

      if (!response.data.success) {
        throw new Error(response.data.error || 'Classification failed');
      }

      return {
        success: true,
        isLanternFly: response.data.is_lantern_fly,
        confidenceScore: response.data.confidence_score,
        pointsAwarded: response.data.points_awarded,
        modelVersion: response.data.model_version,
        classProbabilities: response.data.class_probabilities
      };

    } catch (error) {
      console.error('ML classification error:', error.message);
      
      // Return fallback result if ML service is unavailable
      if (error.code === 'ECONNREFUSED' || error.code === 'ETIMEDOUT') {
        console.warn('ML service unavailable, using fallback classification');
        return this.getFallbackResult();
      }

      throw new Error(`Classification failed: ${error.message}`);
    }
  }

  /**
   * Get fallback classification result when ML service is unavailable
   * @returns {Object} Fallback classification result
   */
  getFallbackResult() {
    // Conservative fallback - assume it's not a lanternfly to avoid false positives
    return {
      success: true,
      isLanternFly: false,
      confidenceScore: 0.5,
      pointsAwarded: 0,
      modelVersion: '1.0.0-fallback',
      classProbabilities: {
        non_lanternfly: 0.5,
        lanternfly: 0.5
      }
    };
  }

  /**
   * Get service status information
   * @returns {Promise<Object>} Service status
   */
  async getStatus() {
    try {
      const response = await axios.get(`${this.mlApiUrl}/health`, {
        timeout: 5000
      });
      return {
        available: true,
        status: response.data.status,
        modelLoaded: response.data.model_loaded
      };
    } catch (error) {
      return {
        available: false,
        error: error.message
      };
    }
  }
}

// Export singleton instance
export const mlService = new MLService();
export default mlService;
