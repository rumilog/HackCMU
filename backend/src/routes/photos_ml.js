import express from 'express';
import multer from 'multer';
import jwt from 'jsonwebtoken';
import { User, Photo } from '../models/index.js';
import { mlService } from '../services/mlService.js';
import { Op } from 'sequelize';
import path from 'path';
import fs from 'fs';

const router = express.Router();

// Configure multer for file uploads
const storage = multer.diskStorage({
  destination: (req, file, cb) => {
    const uploadDir = path.join(process.cwd(), 'uploads');
    console.log('Upload directory:', uploadDir);
    if (!fs.existsSync(uploadDir)) {
      fs.mkdirSync(uploadDir, { recursive: true });
    }
    cb(null, uploadDir);
  },
  filename: (req, file, cb) => {
    const uniqueSuffix = Date.now() + '-' + Math.round(Math.random() * 1E9);
    const filename = file.fieldname + '-' + uniqueSuffix + path.extname(file.originalname);
    console.log('Generated filename:', filename);
    cb(null, filename);
  }
});

const upload = multer({
  storage,
  limits: {
    fileSize: 10 * 1024 * 1024, // 10MB limit
  },
  // Temporarily disable file filter for debugging
  // fileFilter: (req, file, cb) => {
  //   console.log('File filter check:', {
  //     fieldname: file.fieldname,
  //     originalname: file.originalname,
  //     mimetype: file.mimetype,
  //     size: file.size
  //   });
    
  //   if (file.mimetype.startsWith('image/')) {
  //     console.log('File accepted by filter');
  //     cb(null, true);
  //   } else {
  //     console.log('File rejected by filter - not an image');
  //     cb(new Error('Only image files are allowed'), false);
  //   }
  // },
});

// Upload photo endpoint with ML classification
router.post('/upload', upload.single('photo'), async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({
        success: false,
        message: 'No photo uploaded'
      });
    }

    // Get user from token
    const token = req.headers.authorization?.replace('Bearer ', '');
    if (!token) {
      return res.status(401).json({
        success: false,
        message: 'Authentication required'
      });
    }

    const decoded = jwt.verify(token, process.env.JWT_SECRET || 'fallback-secret-key');
    const user = await User.findByPk(decoded.userId);
    if (!user) {
      return res.status(401).json({
        success: false,
        message: 'User not found'
      });
    }

    // Classify the image using ML model
    let classificationResult;
    try {
      classificationResult = await mlService.classifyImage(req.file.path);
      console.log('ML Classification result:', classificationResult);
    } catch (error) {
      console.error('ML classification failed:', error);
      // Use fallback classification if ML service fails
      classificationResult = mlService.getFallbackResult();
    }

    const isLanternFly = classificationResult.isLanternFly;
    const confidence = classificationResult.confidenceScore;
    const pointsAwarded = classificationResult.pointsAwarded;

    // Create photo record in database
    const photo = await Photo.create({
      user_id: user.id,
      image_url: `/uploads/${req.file.filename}`,
      thumbnail_url: `/uploads/${req.file.filename}`, // Same for now
      latitude: req.body.latitude ? parseFloat(req.body.latitude) : null,
      longitude: req.body.longitude ? parseFloat(req.body.longitude) : null,
      location_name: req.body.location_name || null,
      is_lantern_fly: isLanternFly,
      confidence_score: confidence,
      points_awarded: pointsAwarded,
    });

    // Update user stats
    await user.update({
      total_photos: user.total_photos + 1,
      confirmed_lantern_flies: isLanternFly ? user.confirmed_lantern_flies + 1 : user.confirmed_lantern_flies,
      points: user.points + pointsAwarded,
    });


    res.status(201).json({
      success: true,
      message: isLanternFly ? 'Lanternfly detected! Points awarded.' : 'No lanternfly detected.',
      data: {
        id: photo.id,
        image_url: photo.image_url,
        thumbnail_url: photo.thumbnail_url,
        latitude: photo.latitude,
        longitude: photo.longitude,
        location_name: photo.location_name,
        is_lantern_fly: photo.is_lantern_fly,
        confidence_score: photo.confidence_score,
        points_awarded: photo.points_awarded,
        created_at: photo.created_at,
        updated_at: photo.updated_at,
        model_version: classificationResult.modelVersion,
        class_probabilities: classificationResult.classProbabilities
      }
    });

  } catch (error) {
    console.error('Photo upload error:', error);
    
    // Clean up uploaded file if it exists
    if (req.file && fs.existsSync(req.file.path)) {
      fs.unlinkSync(req.file.path);
    }
    
    res.status(500).json({
      success: false,
      message: 'Photo upload failed',
    });
  }
});

// Classify photo endpoint (for testing/standalone classification)
router.post('/classify', (req, res) => {
  upload.single('image')(req, res, async (err) => {
    if (err) {
      console.error('Multer error:', err);
      return res.status(400).json({
        success: false,
        message: err.message || 'File upload error'
      });
    }
    
    try {
      console.log('Classification request received');
      console.log('Request body:', req.body);
      console.log('Request file:', req.file);
      console.log('Request headers:', req.headers);
      
      if (!req.file) {
        console.log('No file received in request');
        return res.status(400).json({
          success: false,
          message: 'No image uploaded'
        });
      }

    // Classify the image using ML model
    let classificationResult;
    try {
      classificationResult = await mlService.classifyImage(req.file.path);
    } catch (error) {
      console.error('ML classification failed:', error);
      // Use fallback classification if ML service fails
      classificationResult = mlService.getFallbackResult();
    }

    // Clean up uploaded file
    if (fs.existsSync(req.file.path)) {
      fs.unlinkSync(req.file.path);
    }

      res.json({
        success: true,
        is_lantern_fly: classificationResult.isLanternFly,
        confidence_score: classificationResult.confidenceScore,
        points_awarded: classificationResult.pointsAwarded,
        model_version: classificationResult.modelVersion,
        class_probabilities: classificationResult.classProbabilities
      });

    } catch (error) {
      console.error('Photo classification error:', error);
      
      // Clean up uploaded file if it exists
      if (req.file && fs.existsSync(req.file.path)) {
        fs.unlinkSync(req.file.path);
      }
      
      res.status(500).json({
        success: false,
        message: 'Photo classification failed',
      });
    }
  });
});

// Get ML service status
router.get('/ml-status', async (req, res) => {
  try {
    const status = await mlService.getStatus();
    res.json({
      success: true,
      ml_service: status
    });
  } catch (error) {
    console.error('ML status check error:', error);
    res.status(500).json({
      success: false,
      message: 'Failed to check ML service status',
    });
  }
});

// Get user photos endpoint
router.get('/user', async (req, res) => {
  try {
    const token = req.headers.authorization?.replace('Bearer ', '');
    if (!token) {
      return res.status(401).json({
        success: false,
        message: 'Authentication required'
      });
    }

    const decoded = jwt.verify(token, process.env.JWT_SECRET || 'fallback-secret-key');
    const user = await User.findByPk(decoded.userId);
    if (!user) {
      return res.status(401).json({
        success: false,
        message: 'User not found'
      });
    }

    const page = parseInt(req.query.page) || 1;
    const limit = parseInt(req.query.limit) || 20;
    const offset = (page - 1) * limit;

    const { count, rows: photos } = await Photo.findAndCountAll({
      where: { user_id: user.id },
      order: [['created_at', 'DESC']],
      limit,
      offset,
    });

    res.json({
      success: true,
      data: {
        photos,
        total: count,
        page,
        limit,
        total_pages: Math.ceil(count / limit)
      }
    });

  } catch (error) {
    console.error('Get user photos error:', error);
    res.status(500).json({
      success: false,
      message: 'Failed to get user photos',
    });
  }
});

// Get research dataset statistics
router.get('/research-dataset-stats', async (req, res) => {
  try {
    const stats = await researchDatasetService.getDatasetStats();
    res.json({
      success: true,
      dataset_stats: stats
    });
  } catch (error) {
    console.error('Research dataset stats error:', error);
    res.status(500).json({
      success: false,
      message: 'Failed to get research dataset statistics',
    });
  }
});

// Get map locations endpoint
router.get('/map', async (req, res) => {
  try {
    // Get all photos with location data for the map
    const photos = await Photo.findAll({
      where: {
        latitude: { [Op.not]: null },
        longitude: { [Op.not]: null }
      },
      include: [{
        model: User,
        as: 'user',
        attributes: ['username']
      }],
      order: [['created_at', 'DESC']],
      limit: 1000 // Limit to prevent too many markers
    });

    // Format the data for the map
    const mapLocations = photos.map(photo => ({
      id: photo.id,
      latitude: photo.latitude,
      longitude: photo.longitude,
      is_lantern_fly: photo.is_lantern_fly,
      confidence_score: photo.confidence_score,
      created_at: photo.created_at,
      username: photo.user?.username || 'Unknown',
      location_name: photo.location_name,
      image_url: photo.image_url
    }));

    res.json(mapLocations);
  } catch (error) {
    console.error('Get map locations error:', error);
    res.status(500).json({
      success: false,
      message: 'Failed to fetch map locations',
    });
  }
});

export default router;
