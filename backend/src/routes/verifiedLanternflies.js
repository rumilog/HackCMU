import express from 'express';
import { Photo, User } from '../models/index.js';

const router = express.Router();

// Get all verified lanternfly sightings for the map table
router.get('/all', async (req, res) => {
  try {
    const page = parseInt(req.query.page) || 1;
    const limit = parseInt(req.query.limit) || 50;
    const offset = (page - 1) * limit;

    // Get all photos that are verified lanternflies
    const { count, rows: photos } = await Photo.findAndCountAll({
      where: { 
        is_lantern_fly: true 
      },
      include: [{
        model: User,
        as: 'user',
        attributes: ['username']
      }],
      order: [['created_at', 'DESC']],
      limit,
      offset,
    });

    // Format the data for the map table
    const verifiedSightings = photos.map(photo => ({
      id: photo.id,
      username: photo.user?.username || 'Unknown',
      latitude: photo.latitude,
      longitude: photo.longitude,
      location_name: photo.location_name,
      image_url: photo.image_url,
      confidence_score: photo.confidence_score,
      points_awarded: photo.points_awarded,
      sighting_date: photo.created_at,
      model_version: photo.model_version
    }));

    res.json({
      success: true,
      data: {
        sightings: verifiedSightings,
        pagination: {
          total: count,
          page,
          limit,
          totalPages: Math.ceil(count / limit)
        }
      }
    });

  } catch (error) {
    console.error('Get verified lanternflies error:', error);
    res.status(500).json({
      success: false,
      message: 'Failed to get verified lanternfly sightings',
      error: error.message
    });
  }
});

// Get verified lanternflies for a specific user
router.get('/user/:userId', async (req, res) => {
  try {
    const { userId } = req.params;
    const page = parseInt(req.query.page) || 1;
    const limit = parseInt(req.query.limit) || 20;
    const offset = (page - 1) * limit;

    const { count, rows: photos } = await Photo.findAndCountAll({
      where: { 
        user_id: userId,
        is_lantern_fly: true 
      },
      include: [{
        model: User,
        as: 'user',
        attributes: ['username']
      }],
      order: [['created_at', 'DESC']],
      limit,
      offset,
    });

    const verifiedSightings = photos.map(photo => ({
      id: photo.id,
      username: photo.user?.username || 'Unknown',
      latitude: photo.latitude,
      longitude: photo.longitude,
      location_name: photo.location_name,
      image_url: photo.image_url,
      confidence_score: photo.confidence_score,
      points_awarded: photo.points_awarded,
      sighting_date: photo.created_at,
      model_version: photo.model_version
    }));

    res.json({
      success: true,
      data: {
        sightings: verifiedSightings,
        pagination: {
          total: count,
          page,
          limit,
          totalPages: Math.ceil(count / limit)
        }
      }
    });

  } catch (error) {
    console.error('Get user verified lanternflies error:', error);
    res.status(500).json({
      success: false,
      message: 'Failed to get user verified lanternfly sightings',
      error: error.message
    });
  }
});

// Get statistics for verified lanternflies
router.get('/stats', async (req, res) => {
  try {
    // Get total count of verified lanternflies
    const totalSightings = await Photo.count({
      where: { is_lantern_fly: true }
    });

    // Get count by user
    const sightingsByUser = await Photo.findAll({
      where: { is_lantern_fly: true },
      include: [{
        model: User,
        as: 'user',
        attributes: ['username']
      }],
      attributes: ['user_id'],
      group: ['user_id', 'user.id', 'user.username']
    });

    // Get recent sightings (last 7 days)
    const sevenDaysAgo = new Date();
    sevenDaysAgo.setDate(sevenDaysAgo.getDate() - 7);
    
    const recentSightings = await Photo.count({
      where: { 
        is_lantern_fly: true,
        created_at: {
          [require('sequelize').Op.gte]: sevenDaysAgo
        }
      }
    });

    res.json({
      success: true,
      data: {
        totalSightings,
        recentSightings,
        usersWithSightings: sightingsByUser.length,
        sightingsByUser: sightingsByUser.map(item => ({
          username: item.user?.username || 'Unknown',
          count: item.dataValues.count || 1
        }))
      }
    });

  } catch (error) {
    console.error('Get verified lanternflies stats error:', error);
    res.status(500).json({
      success: false,
      message: 'Failed to get verified lanternfly statistics',
      error: error.message
    });
  }
});

export default router;
