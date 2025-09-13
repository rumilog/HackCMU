import React from 'react';
import {
  Container,
  Typography,
  Card,
  CardContent,
  Box,
  Avatar,
  Chip,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  LinearProgress,
  Alert,
} from '@mui/material';
import {
  EmojiEvents,
  BugReport,
  PhotoCamera,
  TrendingUp,
  Person,
} from '@mui/icons-material';
import { useQuery } from 'react-query';
import { userService } from '../services/userService';
import { useAuth } from '../hooks/useAuth';
import { LeaderboardEntry } from '../types';

const LeaderboardPage: React.FC = () => {
  const { user } = useAuth();

  const { data: leaderboard = [], isLoading, error } = useQuery(
    'leaderboard',
    userService.getLeaderboard,
    {
      refetchInterval: 60000, // Refetch every minute
    }
  );

  const getRankIcon = (index: number) => {
    switch (index) {
      case 0:
        return '🥇';
      case 1:
        return '🥈';
      case 2:
        return '🥉';
      default:
        return `#${index + 1}`;
    }
  };

  const getRankColor = (index: number) => {
    switch (index) {
      case 0:
        return 'warning';
      case 1:
        return 'default';
      case 2:
        return 'secondary';
      default:
        return 'primary';
    }
  };

  const currentUserRank = leaderboard.findIndex(entry => entry.id === user?.id);

  if (isLoading) {
    return (
      <Container maxWidth="lg" sx={{ py: 4 }}>
        <Typography variant="h4" gutterBottom align="center">
          🏆 Leaderboard
        </Typography>
        <Box sx={{ display: 'flex', justifyContent: 'center', mt: 4 }}>
          <LinearProgress sx={{ width: '100%', maxWidth: 400 }} />
        </Box>
      </Container>
    );
  }

  if (error) {
    return (
      <Container maxWidth="lg" sx={{ py: 4 }}>
        <Typography variant="h4" gutterBottom align="center">
          🏆 Leaderboard
        </Typography>
        <Alert severity="error" sx={{ mt: 2 }}>
          Failed to load leaderboard data. Please try again later.
        </Alert>
      </Container>
    );
  }

  return (
    <Container maxWidth="lg" sx={{ py: 4 }}>
      <Typography variant="h4" gutterBottom align="center">
        🏆 Leaderboard
      </Typography>
      <Typography variant="body1" color="text.secondary" align="center" sx={{ mb: 4 }}>
        Top contributors in the lantern fly tracking community
      </Typography>

      {/* Current User Stats */}
      {currentUserRank !== -1 && (
        <Card sx={{ mb: 4, bgcolor: 'primary.light', color: 'white' }}>
          <CardContent>
            <Typography variant="h6" gutterBottom>
              🎯 Your Ranking
            </Typography>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
              <Avatar sx={{ bgcolor: 'white', color: 'primary.main' }}>
                <Typography variant="h6" fontWeight="bold">
                  {getRankIcon(currentUserRank)}
                </Typography>
              </Avatar>
              <Box>
                <Typography variant="h6">
                  {leaderboard[currentUserRank]?.username}
                </Typography>
                <Typography variant="body2">
                  {leaderboard[currentUserRank]?.points} points • 
                  {leaderboard[currentUserRank]?.confirmed_lantern_flies} lantern flies • 
                  {leaderboard[currentUserRank]?.accuracy_percentage}% accuracy
                </Typography>
              </Box>
            </Box>
          </CardContent>
        </Card>
      )}

      {/* Top 3 Podium */}
      {leaderboard.length >= 3 && (
        <Box sx={{ display: 'flex', justifyContent: 'center', gap: 2, mb: 4 }}>
          {/* 2nd Place */}
          {leaderboard[1] && (
            <Card sx={{ width: 200, textAlign: 'center' }}>
              <CardContent>
                <Typography variant="h2">🥈</Typography>
                <Avatar sx={{ mx: 'auto', mb: 1, bgcolor: 'grey.300' }}>
                  <Person />
                </Avatar>
                <Typography variant="h6" gutterBottom>
                  {leaderboard[1].username}
                </Typography>
                <Typography variant="h4" color="primary">
                  {leaderboard[1].points}
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  points
                </Typography>
              </CardContent>
            </Card>
          )}

          {/* 1st Place */}
          {leaderboard[0] && (
            <Card sx={{ width: 200, textAlign: 'center', transform: 'scale(1.1)' }}>
              <CardContent>
                <Typography variant="h2">🥇</Typography>
                <Avatar sx={{ mx: 'auto', mb: 1, bgcolor: 'warning.main' }}>
                  <EmojiEvents />
                </Avatar>
                <Typography variant="h6" gutterBottom>
                  {leaderboard[0].username}
                </Typography>
                <Typography variant="h4" color="primary">
                  {leaderboard[0].points}
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  points
                </Typography>
              </CardContent>
            </Card>
          )}

          {/* 3rd Place */}
          {leaderboard[2] && (
            <Card sx={{ width: 200, textAlign: 'center' }}>
              <CardContent>
                <Typography variant="h2">🥉</Typography>
                <Avatar sx={{ mx: 'auto', mb: 1, bgcolor: 'secondary.main' }}>
                  <Person />
                </Avatar>
                <Typography variant="h6" gutterBottom>
                  {leaderboard[2].username}
                </Typography>
                <Typography variant="h4" color="primary">
                  {leaderboard[2].points}
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  points
                </Typography>
              </CardContent>
            </Card>
          )}
        </Box>
      )}

      {/* Full Leaderboard Table */}
      <Card>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            📊 Complete Rankings
          </Typography>
          <TableContainer component={Paper} variant="outlined">
            <Table>
              <TableHead>
                <TableRow>
                  <TableCell>Rank</TableCell>
                  <TableCell>User</TableCell>
                  <TableCell align="right">Points</TableCell>
                  <TableCell align="right">Photos</TableCell>
                  <TableCell align="right">Lantern Flies</TableCell>
                  <TableCell align="right">Accuracy</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {leaderboard.map((entry, index) => (
                  <TableRow
                    key={entry.id}
                    sx={{
                      backgroundColor: entry.id === user?.id ? 'primary.light' : 'inherit',
                      '&:hover': {
                        backgroundColor: entry.id === user?.id ? 'primary.main' : 'action.hover',
                      },
                    }}
                  >
                    <TableCell>
                      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                        <Typography variant="h6">
                          {getRankIcon(index)}
                        </Typography>
                        {index < 3 && (
                          <Chip
                            label={`#${index + 1}`}
                            size="small"
                            color={getRankColor(index) as any}
                          />
                        )}
                      </Box>
                    </TableCell>
                    <TableCell>
                      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                        <Avatar sx={{ width: 32, height: 32 }}>
                          {entry.username.charAt(0).toUpperCase()}
                        </Avatar>
                        <Typography variant="body2" fontWeight={entry.id === user?.id ? 'bold' : 'normal'}>
                          {entry.username}
                        </Typography>
                      </Box>
                    </TableCell>
                    <TableCell align="right">
                      <Typography variant="body2" fontWeight="bold">
                        {entry.points}
                      </Typography>
                    </TableCell>
                    <TableCell align="right">
                      <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'flex-end', gap: 0.5 }}>
                        <PhotoCamera fontSize="small" color="action" />
                        <Typography variant="body2">
                          {entry.total_photos}
                        </Typography>
                      </Box>
                    </TableCell>
                    <TableCell align="right">
                      <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'flex-end', gap: 0.5 }}>
                        <BugReport fontSize="small" color="success" />
                        <Typography variant="body2">
                          {entry.confirmed_lantern_flies}
                        </Typography>
                      </Box>
                    </TableCell>
                    <TableCell align="right">
                      <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'flex-end', gap: 1 }}>
                        <Box sx={{ width: 60 }}>
                          <LinearProgress
                            variant="determinate"
                            value={entry.accuracy_percentage}
                            color={entry.accuracy_percentage >= 80 ? 'success' : entry.accuracy_percentage >= 60 ? 'warning' : 'error'}
                            sx={{ height: 6, borderRadius: 3 }}
                          />
                        </Box>
                        <Typography variant="body2" sx={{ minWidth: 40 }}>
                          {entry.accuracy_percentage}%
                        </Typography>
                      </Box>
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </TableContainer>
        </CardContent>
      </Card>

      {/* Motivation Card */}
      <Card sx={{ mt: 3, bgcolor: 'success.light', color: 'white' }}>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            🌟 Keep Going!
          </Typography>
          <Typography variant="body1">
            Every photo you take helps researchers understand and control invasive lantern fly populations. 
            You're making a real difference for the environment!
          </Typography>
        </CardContent>
      </Card>
    </Container>
  );
};

export default LeaderboardPage;
