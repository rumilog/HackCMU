import express from 'express';

const router = express.Router();

// Get user stats endpoint
router.get('/stats', async (req, res) => {
  try {
    // For now, return mock data
    // TODO: Implement actual database query
    const mockStats = {
      total_photos: Math.floor(Math.random() * 50) + 10,
      confirmed_lantern_flies: Math.floor(Math.random() * 20) + 5,
      points: Math.floor(Math.random() * 200) + 50,
      accuracy_percentage: Math.floor(Math.random() * 30) + 70, // 70-100%
    };

    res.json(mockStats);
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
    // For now, return mock data
    // TODO: Implement actual database query
    const mockLeaderboard = Array.from({ length: 10 }, (_, i) => ({
      id: i + 1,
      username: `User${i + 1}`,
      points: Math.floor(Math.random() * 500) + 100,
      total_photos: Math.floor(Math.random() * 100) + 20,
      confirmed_lantern_flies: Math.floor(Math.random() * 50) + 10,
      accuracy_percentage: Math.floor(Math.random() * 30) + 70,
    })).sort((a, b) => b.points - a.points);

    res.json(mockLeaderboard);
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
    // For now, return mock data
    // TODO: Implement actual database query
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
  } catch (error) {
    console.error('Get user achievements error:', error);
    res.status(500).json({
      success: false,
      message: 'Failed to fetch achievements',
    });
  }
});

export default router;
