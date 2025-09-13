import express from 'express';

const router = express.Router();

// Get map data endpoint
router.get('/', async (req, res) => {
  try {
    // For now, return mock data
    // TODO: Implement actual database query
    const mockMapData = Array.from({ length: 15 }, (_, i) => ({
      id: i + 1,
      latitude: 40.7128 + (Math.random() - 0.5) * 0.3,
      longitude: -74.0060 + (Math.random() - 0.5) * 0.3,
      is_lantern_fly: Math.random() > 0.4,
      confidence_score: Math.random() * 0.4 + 0.6,
      created_at: new Date(Date.now() - i * 3600000).toISOString(),
    }));

    res.json(mockMapData);
  } catch (error) {
    console.error('Get map data error:', error);
    res.status(500).json({
      success: false,
      message: 'Failed to fetch map data',
    });
  }
});

export default router;
