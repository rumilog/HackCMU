import React from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Container,
  Grid,
  Card,
  CardContent,
  Typography,
  Button,
  Box,
  LinearProgress,
  Chip,
  Avatar,
} from '@mui/material';
import {
  CameraAlt,
  Map,
  Leaderboard,
  BugReport,
  LocationOn,
  EmojiEvents,
  Refresh,
} from '@mui/icons-material';
import { useAuth } from '../hooks/useAuth';
import { useQuery } from 'react-query';
import { userService } from '../services/userService';

const DashboardPage: React.FC = () => {
  const navigate = useNavigate();
  const { user } = useAuth();

  const { data: stats, isLoading: statsLoading, refetch: refetchStats } = useQuery(
    'userStats',
    userService.getUserStats,
    {
      refetchInterval: 30000, // Refetch every 30 seconds
    }
  );

  const { data: achievements, isLoading: achievementsLoading } = useQuery(
    'userAchievements',
    userService.getUserAchievements,
    {
      refetchInterval: 60000, // Refetch every minute
    }
  );

  const quickActions = [
    {
      title: 'Take Photo',
      description: 'Capture and classify a lantern fly',
      icon: <CameraAlt sx={{ fontSize: 40 }} />,
      color: 'primary',
      action: () => navigate('/camera'),
    },
    {
      title: 'View Map',
      description: 'See reported locations',
      icon: <Map sx={{ fontSize: 40 }} />,
      color: 'secondary',
      action: () => navigate('/map'),
    },
    {
      title: 'Leaderboard',
      description: 'Check your ranking',
      icon: <Leaderboard sx={{ fontSize: 40 }} />,
      color: 'success',
      action: () => navigate('/leaderboard'),
    },
  ];

  const getAccuracyColor = (accuracy: number) => {
    if (accuracy >= 80) return 'success';
    if (accuracy >= 60) return 'warning';
    return 'error';
  };

  return (
    <Container maxWidth="lg" sx={{ py: 4 }}>
      {/* Welcome Section */}
      <Box sx={{ mb: 4, display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
        <Box>
          <Typography variant="h4" gutterBottom>
            Welcome back, {user?.username}! 🦟
          </Typography>
          <Typography variant="body1" color="text.secondary">
            Help track invasive lantern fly populations in your area
          </Typography>
        </Box>
        <Button
          variant="outlined"
          startIcon={<Refresh />}
          onClick={() => refetchStats()}
          disabled={statsLoading}
          sx={{ 
            borderColor: 'primary.main',
            color: 'primary.main',
            '&:hover': {
              borderColor: 'primary.dark',
              backgroundColor: 'primary.light',
              color: 'primary.dark'
            }
          }}
        >
          Refresh Stats
        </Button>
      </Box>

      {/* Stats Cards */}
      <Grid container spacing={3} sx={{ mb: 4 }}>
        <Grid item xs={12} sm={6} md={4}>
          <Card>
            <CardContent>
              <Box display="flex" alignItems="center" justifyContent="space-between">
                <Box>
                  <Typography color="text.secondary" gutterBottom>
                    Total Points
                  </Typography>
                  <Typography variant="h4">
                    {statsLoading ? '...' : (stats?.points || 0)}
                  </Typography>
                </Box>
                <Avatar sx={{ bgcolor: 'primary.main' }}>
                  <EmojiEvents />
                </Avatar>
              </Box>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} sm={6} md={4}>
          <Card>
            <CardContent>
              <Box display="flex" alignItems="center" justifyContent="space-between">
                <Box>
                  <Typography color="text.secondary" gutterBottom>
                    Lantern Flies
                  </Typography>
                  <Typography variant="h4">
                    {statsLoading ? '...' : (stats?.confirmed_lantern_flies || 0)}
                  </Typography>
                </Box>
                <Avatar sx={{ bgcolor: 'success.main' }}>
                  <BugReport />
                </Avatar>
              </Box>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} sm={6} md={4}>
          <Card>
            <CardContent>
              <Box display="flex" alignItems="center" justifyContent="space-between">
                <Box>
                  <Typography color="text.secondary" gutterBottom>
                    Accuracy
                  </Typography>
                  <Typography variant="h4">
                    {statsLoading ? '...' : (stats?.accuracy_percentage || 0)}%
                  </Typography>
                </Box>
                <Avatar sx={{ bgcolor: 'info.main' }}>
                  <LocationOn />
                </Avatar>
              </Box>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Accuracy Progress */}
      {stats && (
        <Card sx={{ mb: 4 }}>
          <CardContent>
            <Typography variant="h6" gutterBottom>
              Classification Accuracy
            </Typography>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
              <LinearProgress
                variant="determinate"
                value={stats.accuracy_percentage}
                sx={{ flexGrow: 1, height: 8, borderRadius: 4 }}
                color={getAccuracyColor(stats.accuracy_percentage)}
              />
              <Typography variant="body2" color="text.secondary">
                {stats.accuracy_percentage}%
              </Typography>
            </Box>
          </CardContent>
        </Card>
      )}

      {/* Quick Actions */}
      <Typography variant="h5" gutterBottom sx={{ mb: 3 }}>
        Quick Actions
      </Typography>
      <Grid container spacing={3} sx={{ mb: 4 }}>
        {quickActions.map((action, index) => (
          <Grid item xs={12} sm={6} md={4} key={index}>
            <Card
              sx={{
                cursor: 'pointer',
                transition: 'transform 0.2s',
                '&:hover': {
                  transform: 'translateY(-4px)',
                },
              }}
              onClick={action.action}
            >
              <CardContent sx={{ textAlign: 'center', py: 3 }}>
                <Box sx={{ color: `${action.color}.main`, mb: 2 }}>
                  {action.icon}
                </Box>
                <Typography variant="h6" gutterBottom>
                  {action.title}
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  {action.description}
                </Typography>
              </CardContent>
            </Card>
          </Grid>
        ))}
      </Grid>

      {/* Recent Achievements */}
      {achievements && achievements.length > 0 && (
        <Card>
          <CardContent>
            <Typography variant="h6" gutterBottom>
              Recent Achievements
            </Typography>
            <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1 }}>
              {achievements.slice(0, 5).map((achievement) => (
                <Chip
                  key={achievement.id}
                  label={achievement.achievement_name}
                  color="primary"
                  variant="outlined"
                  icon={<EmojiEvents />}
                />
              ))}
            </Box>
          </CardContent>
        </Card>
      )}

      {/* Environmental Impact */}
      <Card sx={{ mt: 4, bgcolor: 'success.light', color: 'white' }}>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            🌱 Environmental Impact
          </Typography>
          <Typography variant="body1">
            By tracking lantern flies, you're helping researchers understand and control 
            this invasive species. Every photo contributes to scientific research and 
            environmental protection efforts.
          </Typography>
        </CardContent>
      </Card>
    </Container>
  );
};

export default DashboardPage;
