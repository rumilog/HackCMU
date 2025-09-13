import express from 'express';
import cors from 'cors';
import helmet from 'helmet';
import compression from 'compression';
import morgan from 'morgan';
import rateLimit from 'express-rate-limit';
import dotenv from 'dotenv';
import bcrypt from 'bcryptjs';
import jwt from 'jsonwebtoken';

// Load environment variables
dotenv.config();

const app = express();
const PORT = process.env.PORT || 5000;

// In-memory storage for demo purposes
const users = [];
let nextUserId = 1;

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
    const existingUser = users.find(u => u.username === username);
    if (existingUser) {
      return res.status(400).json({
        success: false,
        message: 'Username already exists'
      });
    }

    // Check if email already exists (if provided)
    if (email) {
      const existingEmail = users.find(u => u.email === email);
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

    // Create user
    const user = {
      id: nextUserId++,
      username,
      password_hash: passwordHash,
      email: email || null,
      points: 0,
      total_photos: 0,
      confirmed_lantern_flies: 0,
      created_at: new Date().toISOString(),
      updated_at: new Date().toISOString(),
    };

    users.push(user);

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

    // Find user
    const user = users.find(u => u.username === username);
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
    
    // Find user
    const user = users.find(u => u.id === decoded.userId);
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

// Mock photo routes
app.post('/api/photos/upload', (req, res) => {
  res.json({
    success: true,
    message: 'Photo uploaded successfully (mock)',
    data: {
      id: Math.floor(Math.random() * 1000),
      user_id: 1,
      image_url: 'https://via.placeholder.com/400x300/4CAF50/white?text=Mock+Photo',
      is_lantern_fly: Math.random() > 0.5,
      confidence_score: Math.random(),
      points_awarded: Math.random() > 0.5 ? 10 : 1,
      created_at: new Date().toISOString(),
    }
  });
});

app.post('/api/photos/classify', (req, res) => {
  const isLanternFly = Math.random() > 0.3;
  const confidence = Math.random() * 0.4 + 0.6;
  const pointsAwarded = isLanternFly ? 10 : 1;

  res.json({
    success: true,
    is_lantern_fly: isLanternFly,
    confidence_score: confidence,
    points_awarded: pointsAwarded,
    model_version: '1.0.0-mock',
  });
});

app.get('/api/photos/user', (req, res) => {
  const mockPhotos = Array.from({ length: 5 }, (_, i) => ({
    id: i + 1,
    user_id: 1,
    image_url: `https://via.placeholder.com/400x300/4CAF50/white?text=Photo+${i + 1}`,
    is_lantern_fly: Math.random() > 0.5,
    confidence_score: Math.random(),
    points_awarded: Math.random() > 0.5 ? 10 : 1,
    created_at: new Date(Date.now() - i * 86400000).toISOString(),
  }));

  res.json({
    success: true,
    photos: mockPhotos,
    total: 25,
  });
});

app.get('/api/photos/map', (req, res) => {
  const mockLocations = Array.from({ length: 10 }, (_, i) => ({
    id: i + 1,
    latitude: 40.7128 + (Math.random() - 0.5) * 0.2,
    longitude: -74.0060 + (Math.random() - 0.5) * 0.2,
    is_lantern_fly: Math.random() > 0.4,
    confidence_score: Math.random() * 0.4 + 0.6,
    created_at: new Date(Date.now() - i * 3600000).toISOString(),
  }));

  res.json(mockLocations);
});

// Mock user routes
app.get('/api/users/stats', (req, res) => {
  res.json({
    total_photos: Math.floor(Math.random() * 50) + 10,
    confirmed_lantern_flies: Math.floor(Math.random() * 20) + 5,
    points: Math.floor(Math.random() * 200) + 50,
    accuracy_percentage: Math.floor(Math.random() * 30) + 70,
  });
});

app.get('/api/users/leaderboard', (req, res) => {
  const mockLeaderboard = Array.from({ length: 10 }, (_, i) => ({
    id: i + 1,
    username: `User${i + 1}`,
    points: Math.floor(Math.random() * 500) + 100,
    total_photos: Math.floor(Math.random() * 100) + 20,
    confirmed_lantern_flies: Math.floor(Math.random() * 50) + 10,
    accuracy_percentage: Math.floor(Math.random() * 30) + 70,
  })).sort((a, b) => b.points - a.points);

  res.json(mockLeaderboard);
});

app.get('/api/users/achievements', (req, res) => {
  const mockAchievements = [
    {
      id: 1,
      user_id: 1,
      achievement_type: 'first_photo',
      achievement_name: 'First Spotter',
      description: 'Uploaded your first photo',
      points_reward: 5,
      earned_at: new Date(Date.now() - 86400000).toISOString(),
    },
    {
      id: 2,
      user_id: 1,
      achievement_type: 'lantern_fly_detection',
      achievement_name: 'Lantern Fly Hunter',
      description: 'Detected your first lantern fly',
      points_reward: 25,
      earned_at: new Date(Date.now() - 172800000).toISOString(),
    },
  ];

  res.json(mockAchievements);
});

// Root endpoint
app.get('/', (req, res) => {
  res.json({
    message: 'Lantern Fly Tracker API',
    version: '1.0.0',
    endpoints: {
      auth: '/api/auth',
      photos: '/api/photos',
      users: '/api/users',
    }
  });
});

// Start server
app.listen(PORT, () => {
  console.log(`🚀 Server running on port ${PORT}`);
  console.log(`📊 Health check: http://localhost:${PORT}/health`);
  console.log(`🌍 Environment: ${process.env.NODE_ENV || 'development'}`);
  console.log(`🔗 API Base URL: http://localhost:${PORT}/api`);
  console.log(`👥 Users stored in memory: ${users.length}`);
});

export default app;
