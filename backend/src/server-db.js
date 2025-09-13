import express from 'express';
import cors from 'cors';
import helmet from 'helmet';
import compression from 'compression';
import morgan from 'morgan';
import rateLimit from 'express-rate-limit';
import dotenv from 'dotenv';
import bcrypt from 'bcryptjs';
import jwt from 'jsonwebtoken';
import multer from 'multer';
import path from 'path';
import { fileURLToPath } from 'url';

// Import database and models
import { testConnection, initializeDatabase } from './config/database.js';
import { User, Photo } from './models/index.js';

// Load environment variables
dotenv.config();

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
const PORT = process.env.PORT || 5000;

// Configure multer for file uploads
const storage = multer.diskStorage({
  destination: (req, file, cb) => {
    cb(null, path.join(__dirname, '../uploads/'));
  },
  filename: (req, file, cb) => {
    const uniqueSuffix = Date.now() + '-' + Math.round(Math.random() * 1E9);
    cb(null, file.fieldname + '-' + uniqueSuffix + path.extname(file.originalname));
  }
});

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

// Create uploads directory if it doesn't exist
import fs from 'fs';
const uploadsDir = path.join(__dirname, '../uploads');
if (!fs.existsSync(uploadsDir)) {
  fs.mkdirSync(uploadsDir, { recursive: true });
}

// Security middleware
app.use(helmet());
app.use(cors({
  origin: process.env.CORS_ORIGIN || 'http://localhost:3000',
  credentials: true
}));

// Rate limiting
const limiter = rateLimit({
  windowMs: 15 * 60 * 1000, // 15 minutes
  max: 100, // limit each IP to 100 requests per windowMs
  message: 'Too many requests from this IP, please try again later.'
});
app.use('/api/', limiter);

// Body parsing middleware
app.use(express.json({ limit: '10mb' }));
app.use(express.urlencoded({ extended: true, limit: '10mb' }));

// Compression middleware
app.use(compression());

// Logging middleware
app.use(morgan('combined'));

// Serve uploaded files
app.use('/uploads', express.static(path.join(__dirname, '../uploads')));

// Health check endpoint
app.get('/health', (req, res) => {
  res.status(200).json({
    status: 'OK',
    timestamp: new Date().toISOString(),
    uptime: process.uptime()
  });
});

// Auth routes
app.post('/api/auth/register', async (req, res) => {
  try {
    const { username, password, email } = req.body;

    // Validation
    if (!username || username.length < 3) {
      return res.status(400).json({
        success: false,
        message: 'Username must be at least 3 characters long'
      });
    }

    if (!password || password.length < 6) {
      return res.status(400).json({
        success: false,
        message: 'Password must be at least 6 characters long'
      });
    }

    // Check if user already exists
    const existingUser = await User.findOne({ where: { username } });
    if (existingUser) {
      return res.status(400).json({
        success: false,
        message: 'Username already exists'
      });
    }

    // Check if email already exists (if provided)
    if (email) {
      const existingEmail = await User.findOne({ where: { email } });
      if (existingEmail) {
        return res.status(400).json({
          success: false,
          message: 'Email already exists'
        });
      }
    }

    // Hash password
    const saltRounds = 12;
    const passwordHash = await bcrypt.hash(password, saltRounds);

    // Create user in database
    const user = await User.create({
      username,
      password_hash: passwordHash,
      email: email || null,
      points: 0,
      total_photos: 0,
      confirmed_lantern_flies: 0,
    });

    // Generate JWT token
    const token = jwt.sign(
      { userId: user.id, username: user.username },
      process.env.JWT_SECRET || 'fallback-secret-key',
      { expiresIn: process.env.JWT_EXPIRES_IN || '7d' }
    );

    // Return user data (without password)
    const userResponse = {
      id: user.id,
      username: user.username,
      email: user.email,
      points: user.points,
      total_photos: user.total_photos,
      confirmed_lantern_flies: user.confirmed_lantern_flies,
      created_at: user.created_at,
      updated_at: user.updated_at,
    };

    res.status(201).json({
      success: true,
      message: 'User registered successfully',
      token,
      user: userResponse,
    });
  } catch (error) {
    console.error('Registration error:', error);
    res.status(500).json({
      success: false,
      message: 'Internal server error',
    });
  }
});

app.post('/api/auth/login', async (req, res) => {
  try {
    const { username, password } = req.body;

    // Validation
    if (!username || !password) {
      return res.status(400).json({
        success: false,
        message: 'Username and password are required'
      });
    }

    // Find user in database
    const user = await User.findOne({ where: { username } });
    if (!user) {
      return res.status(401).json({
        success: false,
        message: 'Invalid username or password'
      });
    }

    // Check password
    const isValidPassword = await bcrypt.compare(password, user.password_hash);
    if (!isValidPassword) {
      return res.status(401).json({
        success: false,
        message: 'Invalid username or password'
      });
    }

    // Generate JWT token
    const token = jwt.sign(
      { userId: user.id, username: user.username },
      process.env.JWT_SECRET || 'fallback-secret-key',
      { expiresIn: process.env.JWT_EXPIRES_IN || '7d' }
    );

    // Return user data (without password)
    const userResponse = {
      id: user.id,
      username: user.username,
      email: user.email,
      points: user.points,
      total_photos: user.total_photos,
      confirmed_lantern_flies: user.confirmed_lantern_flies,
      created_at: user.created_at,
      updated_at: user.updated_at,
    };

    res.json({
      success: true,
      message: 'Login successful',
      token,
      user: userResponse,
    });
  } catch (error) {
    console.error('Login error:', error);
    res.status(500).json({
      success: false,
      message: 'Internal server error',
    });
  }
});

app.get('/api/auth/me', async (req, res) => {
  try {
    const token = req.headers.authorization?.replace('Bearer ', '');
    
    if (!token) {
      return res.status(401).json({
        success: false,
        message: 'No token provided'
      });
    }

    // Verify token
    const decoded = jwt.verify(token, process.env.JWT_SECRET || 'fallback-secret-key');
    
    // Find user in database
    const user = await User.findByPk(decoded.userId);
    if (!user) {
      return res.status(401).json({
        success: false,
        message: 'User not found'
      });
    }

    // Return user data (without password)
    const userResponse = {
      id: user.id,
      username: user.username,
      email: user.email,
      points: user.points,
      total_photos: user.total_photos,
      confirmed_lantern_flies: user.confirmed_lantern_flies,
      created_at: user.created_at,
      updated_at: user.updated_at,
    };

    res.json(userResponse);
  } catch (error) {
    console.error('Get user error:', error);
    res.status(401).json({
      success: false,
      message: 'Invalid token'
    });
  }
});

// Photo routes
app.post('/api/photos/upload', upload.single('photo'), async (req, res) => {
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

    // For now, use mock classification (will be replaced with ML model)
    const isLanternFly = Math.random() > 0.3; // 70% chance of being a lantern fly
    const confidence = Math.random() * 0.4 + 0.6; // 60-100% confidence
    const pointsAwarded = isLanternFly ? 10 : 1;

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
      message: 'Photo uploaded successfully',
      data: {
        id: photo.id,
        user_id: photo.user_id,
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
      }
    });
  } catch (error) {
    console.error('Photo upload error:', error);
    res.status(500).json({
      success: false,
      message: 'Photo upload failed',
    });
  }
});

app.post('/api/photos/classify', upload.single('image'), async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({
        success: false,
        message: 'No image uploaded'
      });
    }

    // For now, return mock classification
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

app.get('/api/photos/user', async (req, res) => {
  try {
    const token = req.headers.authorization?.replace('Bearer ', '');
    if (!token) {
      return res.status(401).json({
        success: false,
        message: 'Authentication required'
      });
    }

    const decoded = jwt.verify(token, process.env.JWT_SECRET || 'fallback-secret-key');
    const page = parseInt(req.query.page) || 1;
    const limit = parseInt(req.query.limit) || 20;
    const offset = (page - 1) * limit;

    // Get user's photos from database
    const { count, rows: photos } = await Photo.findAndCountAll({
      where: { user_id: decoded.userId },
      order: [['created_at', 'DESC']],
      limit,
      offset,
    });

    res.json({
      success: true,
      photos,
      total: count,
      page,
      limit,
      totalPages: Math.ceil(count / limit),
    });
  } catch (error) {
    console.error('Get user photos error:', error);
    res.status(500).json({
      success: false,
      message: 'Failed to fetch photos',
    });
  }
});

app.get('/api/photos/map', async (req, res) => {
  try {
    // Get all photos with location data from database
    const photos = await Photo.findAll({
      where: {
        latitude: { [require('sequelize').Op.ne]: null },
        longitude: { [require('sequelize').Op.ne]: null },
      },
      attributes: ['id', 'latitude', 'longitude', 'is_lantern_fly', 'confidence_score', 'created_at'],
      order: [['created_at', 'DESC']],
    });

    res.json(photos);
  } catch (error) {
    console.error('Get map locations error:', error);
    res.status(500).json({
      success: false,
      message: 'Failed to fetch map locations',
    });
  }
});

// User routes
app.get('/api/users/stats', async (req, res) => {
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
      return res.status(404).json({
        success: false,
        message: 'User not found'
      });
    }

    const accuracy = user.total_photos > 0 
      ? Math.round((user.confirmed_lantern_flies / user.total_photos) * 100)
      : 0;

    res.json({
      total_photos: user.total_photos,
      confirmed_lantern_flies: user.confirmed_lantern_flies,
      points: user.points,
      accuracy_percentage: accuracy,
    });
  } catch (error) {
    console.error('Get user stats error:', error);
    res.status(500).json({
      success: false,
      message: 'Failed to fetch user stats',
    });
  }
});

app.get('/api/users/leaderboard', async (req, res) => {
  try {
    // Get top users from database
    const users = await User.findAll({
      attributes: ['id', 'username', 'points', 'total_photos', 'confirmed_lantern_flies'],
      order: [['points', 'DESC']],
      limit: 50,
    });

    const leaderboard = users.map(user => ({
      id: user.id,
      username: user.username,
      points: user.points,
      total_photos: user.total_photos,
      confirmed_lantern_flies: user.confirmed_lantern_flies,
      accuracy_percentage: user.total_photos > 0 
        ? Math.round((user.confirmed_lantern_flies / user.total_photos) * 100)
        : 0,
    }));

    res.json(leaderboard);
  } catch (error) {
    console.error('Get leaderboard error:', error);
    res.status(500).json({
      success: false,
      message: 'Failed to fetch leaderboard',
    });
  }
});

app.get('/api/users/achievements', async (req, res) => {
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
      return res.status(404).json({
        success: false,
        message: 'User not found'
      });
    }

    // Generate achievements based on user stats
    const achievements = [];
    
    if (user.total_photos > 0) {
      achievements.push({
        id: 1,
        user_id: user.id,
        achievement_type: 'first_photo',
        achievement_name: 'First Spotter',
        description: 'Uploaded your first photo',
        points_reward: 5,
        earned_at: user.created_at,
      });
    }

    if (user.confirmed_lantern_flies > 0) {
      achievements.push({
        id: 2,
        user_id: user.id,
        achievement_type: 'lantern_fly_detection',
        achievement_name: 'Lantern Fly Hunter',
        description: 'Detected your first lantern fly',
        points_reward: 25,
        earned_at: user.created_at,
      });
    }

    if (user.points >= 100) {
      achievements.push({
        id: 3,
        user_id: user.id,
        achievement_type: 'points_milestone',
        achievement_name: 'Century Club',
        description: 'Earned 100 points',
        points_reward: 0,
        earned_at: user.updated_at,
      });
    }

    res.json(achievements);
  } catch (error) {
    console.error('Get user achievements error:', error);
    res.status(500).json({
      success: false,
      message: 'Failed to fetch achievements',
    });
  }
});

// Root endpoint
app.get('/', (req, res) => {
  res.json({
    message: 'Lantern Fly Tracker API',
    version: '1.0.0',
    database: 'SQLite',
    endpoints: {
      auth: '/api/auth',
      photos: '/api/photos',
      users: '/api/users',
    }
  });
});

// Initialize database and start server
const startServer = async () => {
  try {
    // Test database connection
    const dbConnected = await testConnection();
    if (!dbConnected) {
      console.log('⚠️  Database connection failed, but continuing...');
    } else {
      // Initialize database tables
      await initializeDatabase();
    }

    // Start server
    app.listen(PORT, () => {
      console.log(`🚀 Server running on port ${PORT}`);
      console.log(`📊 Health check: http://localhost:${PORT}/health`);
      console.log(`🌍 Environment: ${process.env.NODE_ENV || 'development'}`);
      console.log(`🔗 API Base URL: http://localhost:${PORT}/api`);
      console.log(`💾 Database: SQLite`);
      console.log(`📁 Uploads: ${uploadsDir}`);
    });
  } catch (error) {
    console.error('❌ Failed to start server:', error);
    process.exit(1);
  }
};

startServer();

export default app;
