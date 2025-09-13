import express from 'express';
import multer from 'multer';
import { body, validationResult } from 'express-validator';

const router = express.Router();

// Configure multer for file uploads
const storage = multer.memoryStorage();
const upload = multer({
  storage,
  limits: {
    fileSize: 10 * 1024 * 1024, // 10MB limit
  },
  fileFilter: (req, file, cb) => {
    if (file.mimetype.startsWith('image/')) {
      cb(null, true);
    } else {
      cb(new Error('Only image files are allowed'), false);
    }
  },
});

// Upload photo endpoint
router.post('/upload', upload.single('photo'), async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({
        success: false,
        message: 'No photo uploaded',
      });
    }

    // For now, return a mock response
    // TODO: Implement actual photo processing and storage
    const mockPhoto = {
      id: Math.floor(Math.random() * 1000),
      user_id: 1, // TODO: Get from JWT token
      image_url: 'https://via.placeholder.com/400x300/4CAF50/white?text=Mock+Photo',
      thumbnail_url: 'https://via.placeholder.com/200x150/4CAF50/white?text=Thumb',
      latitude: parseFloat(req.body.latitude) || null,
      longitude: parseFloat(req.body.longitude) || null,
      location_name: req.body.location_name || null,
      is_lantern_fly: Math.random() > 0.5, // Random for now
      confidence_score: Math.random(),
      points_awarded: Math.random() > 0.5 ? 10 : 1,
      created_at: new Date().toISOString(),
      updated_at: new Date().toISOString(),
    };

    res.status(201).json({
      success: true,
      message: 'Photo uploaded successfully',
      data: mockPhoto,
    });
  } catch (error) {
    console.error('Photo upload error:', error);
    res.status(500).json({
      success: false,
      message: 'Photo upload failed',
    });
  }
});

// Classify photo endpoint
router.post('/classify', upload.single('image'), async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({
        success: false,
        message: 'No image uploaded',
      });
    }

    // For now, return a mock classification
    // TODO: Integrate with ML model
    const isLanternFly = Math.random() > 0.3; // 70% chance of being a lantern fly
    const confidence = Math.random() * 0.4 + 0.6; // 60-100% confidence
    const pointsAwarded = isLanternFly ? 10 : 1;

    res.json({
      success: true,
      is_lantern_fly: isLanternFly,
      confidence_score: confidence,
      points_awarded: pointsAwarded,
      model_version: '1.0.0-mock',
    });
  } catch (error) {
    console.error('Photo classification error:', error);
    res.status(500).json({
      success: false,
      message: 'Photo classification failed',
    });
  }
});

// Get user photos endpoint
router.get('/user', async (req, res) => {
  try {
    const page = parseInt(req.query.page) || 1;
    const limit = parseInt(req.query.limit) || 20;
    const offset = (page - 1) * limit;

    // For now, return mock data
    // TODO: Implement actual database query
    const mockPhotos = Array.from({ length: Math.min(limit, 5) }, (_, i) => ({
      id: i + 1,
      user_id: 1,
      image_url: `https://via.placeholder.com/400x300/4CAF50/white?text=Photo+${i + 1}`,
      thumbnail_url: `https://via.placeholder.com/200x150/4CAF50/white?text=Thumb+${i + 1}`,
      latitude: 40.7128 + (Math.random() - 0.5) * 0.1,
      longitude: -74.0060 + (Math.random() - 0.5) * 0.1,
      location_name: `Location ${i + 1}`,
      is_lantern_fly: Math.random() > 0.5,
      confidence_score: Math.random(),
      points_awarded: Math.random() > 0.5 ? 10 : 1,
      created_at: new Date(Date.now() - i * 86400000).toISOString(),
      updated_at: new Date(Date.now() - i * 86400000).toISOString(),
    }));

    res.json({
      success: true,
      photos: mockPhotos,
      total: 25, // Mock total
      page,
      limit,
      totalPages: Math.ceil(25 / limit),
    });
  } catch (error) {
    console.error('Get user photos error:', error);
    res.status(500).json({
      success: false,
      message: 'Failed to fetch photos',
    });
  }
});

// Get map locations endpoint
router.get('/map', async (req, res) => {
  try {
    // For now, return mock map data
    // TODO: Implement actual database query
    const mockLocations = Array.from({ length: 10 }, (_, i) => ({
      id: i + 1,
      latitude: 40.7128 + (Math.random() - 0.5) * 0.2,
      longitude: -74.0060 + (Math.random() - 0.5) * 0.2,
      is_lantern_fly: Math.random() > 0.4,
      confidence_score: Math.random() * 0.4 + 0.6,
      created_at: new Date(Date.now() - i * 3600000).toISOString(),
    }));

    res.json(mockLocations);
  } catch (error) {
    console.error('Get map locations error:', error);
    res.status(500).json({
      success: false,
      message: 'Failed to fetch map locations',
    });
  }
});

// Get photo by ID endpoint
router.get('/:id', async (req, res) => {
  try {
    const { id } = req.params;

    // For now, return mock data
    // TODO: Implement actual database query
    const mockPhoto = {
      id: parseInt(id),
      user_id: 1,
      image_url: `https://via.placeholder.com/400x300/4CAF50/white?text=Photo+${id}`,
      thumbnail_url: `https://via.placeholder.com/200x150/4CAF50/white?text=Thumb+${id}`,
      latitude: 40.7128 + (Math.random() - 0.5) * 0.1,
      longitude: -74.0060 + (Math.random() - 0.5) * 0.1,
      location_name: `Location ${id}`,
      is_lantern_fly: Math.random() > 0.5,
      confidence_score: Math.random(),
      points_awarded: Math.random() > 0.5 ? 10 : 1,
      created_at: new Date().toISOString(),
      updated_at: new Date().toISOString(),
    };

    res.json(mockPhoto);
  } catch (error) {
    console.error('Get photo error:', error);
    res.status(500).json({
      success: false,
      message: 'Failed to fetch photo',
    });
  }
});

// Delete photo endpoint
router.delete('/:id', async (req, res) => {
  try {
    const { id } = req.params;

    // For now, return success
    // TODO: Implement actual database deletion
    res.json({
      success: true,
      message: 'Photo deleted successfully',
    });
  } catch (error) {
    console.error('Delete photo error:', error);
    res.status(500).json({
      success: false,
      message: 'Failed to delete photo',
    });
  }
});

export default router;
