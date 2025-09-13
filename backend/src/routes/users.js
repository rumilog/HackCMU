import express from 'express';
import jwt from 'jsonwebtoken';
import { User, Photo } from '../models/index.js';

const router = express.Router();

// Get current user endpoint
router.get('/me', async (req, res) => {
  try {
    // Get user from JWT token
    const token = req.headers.authorization?.split(' ')[1];
    if (!token) {
      return res.status(401).json({
        success: false,
        message: 'No token provided',
      });
    }

    const decoded = jwt.verify(token, process.env.JWT_SECRET || 'fallback-secret');
    const user = await User.findByPk(decoded.userId, {
      attributes: ['id', 'username', 'email', 'points', 'total_photos', 'confirmed_lantern_flies', 'created_at'],
    });
    
    if (!user) {
      return res.status(404).json({
        success: false,
        message: 'User not found',
      });
    }

    res.json(user);
  } catch (error) {
    console.error('Get current user error:', error);
    res.status(500).json({
      success: false,
      message: 'Failed to fetch user data',
    });
  }
});

// Get user stats endpoint
router.get('/stats', async (req, res) => {
  try {
    // Get user from JWT token
    const token = req.headers.authorization?.split(' ')[1];
    if (!token) {
      return res.status(401).json({
        success: false,
        message: 'No token provided',
      });
    }

    const decoded = jwt.verify(token, process.env.JWT_SECRET || 'fallback-secret');
    const user = await User.findByPk(decoded.userId);
    
    if (!user) {
      return res.status(404).json({
        success: false,
        message: 'User not found',
      });
    }

    // Calculate accuracy percentage
    const accuracy_percentage = user.total_photos > 0 
      ? Math.round((user.confirmed_lantern_flies / user.total_photos) * 100)
      : 0;

    const stats = {
      total_photos: user.total_photos,
      confirmed_lantern_flies: user.confirmed_lantern_flies,
      points: user.points,
      accuracy_percentage,
    };

    res.json(stats);
  } catch (error) {
    console.error('Get user stats error:', error);
    res.status(500).json({
      success: false,
      message: 'Failed to fetch user stats',
    });
  }
});

// Get leaderboard endpoint
router.get('/leaderboard', async (req, res) => {
  try {
    // Get real users from database, ordered by points
    const users = await User.findAll({
      attributes: ['id', 'username', 'points', 'total_photos', 'confirmed_lantern_flies'],
      order: [['points', 'DESC']],
      limit: 50, // Top 50 users
    });

    // Transform to leaderboard format
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

// Get user achievements endpoint
router.get('/achievements', async (req, res) => {
  try {
    // For now, return empty array since we don't have achievements table yet
    // TODO: Implement actual achievements system with database
    const achievements = [];

    res.json(achievements);
  } catch (error) {
    console.error('Get user achievements error:', error);
    res.status(500).json({
      success: false,
      message: 'Failed to fetch achievements',
    });
  }
});

export default router;
