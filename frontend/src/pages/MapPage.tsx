import React, { useEffect, useState } from 'react';
import { MapContainer, TileLayer, Marker, Popup, useMap } from 'react-leaflet';
import { 
  Container, 
  Typography, 
  Box, 
  Card, 
  CardContent, 
  Chip, 
  Alert,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  Avatar,
  Pagination,
  CircularProgress
} from '@mui/material';
import { BugReport, LocationOn, CheckCircle, Cancel, Image as ImageIcon } from '@mui/icons-material';
import { Icon } from 'leaflet';
import 'leaflet/dist/leaflet.css';
import { useQuery } from 'react-query';
import { photoService } from '../services/photoService';
import { verifiedLanternfliesService } from '../services/verifiedLanternfliesService';
import { MapLocation } from '../types';

// Fix for default markers in react-leaflet
delete (Icon.Default.prototype as any)._getIconUrl;
Icon.Default.mergeOptions({
  iconRetinaUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-icon-2x.png',
  iconUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-icon.png',
  shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-shadow.png',
});

// Custom icons - using simple colored circles instead of emojis
const lanternFlyIcon = new Icon({
  iconUrl: 'data:image/svg+xml;base64,' + btoa(`
    <svg width="25" height="41" viewBox="0 0 25 41" xmlns="http://www.w3.org/2000/svg">
      <path d="M12.5 0C5.6 0 0 5.6 0 12.5c0 12.5 12.5 28.5 12.5 28.5s12.5-16 12.5-28.5C25 5.6 19.4 0 12.5 0z" fill="#4CAF50"/>
      <circle cx="12.5" cy="12.5" r="8" fill="white"/>
      <circle cx="12.5" cy="12.5" r="5" fill="#4CAF50"/>
    </svg>
  `),
  iconSize: [25, 41],
  iconAnchor: [12, 41],
  popupAnchor: [1, -34],
});

const notLanternFlyIcon = new Icon({
  iconUrl: 'data:image/svg+xml;base64,' + btoa(`
    <svg width="25" height="41" viewBox="0 0 25 41" xmlns="http://www.w3.org/2000/svg">
      <path d="M12.5 0C5.6 0 0 5.6 0 12.5c0 12.5 12.5 28.5 12.5 28.5s12.5-16 12.5-28.5C25 5.6 19.4 0 12.5 0z" fill="#757575"/>
      <circle cx="12.5" cy="12.5" r="8" fill="white"/>
      <circle cx="12.5" cy="12.5" r="5" fill="#757575"/>
    </svg>
  `),
  iconSize: [25, 41],
  iconAnchor: [12, 41],
  popupAnchor: [1, -34],
});

const MapStats: React.FC<{ locations: MapLocation[] }> = ({ locations }) => {
  const totalReports = locations.length;
  const lanternFlyReports = locations.filter(loc => loc.is_lantern_fly).length;
  const notLanternFlyReports = totalReports - lanternFlyReports;

  return (
    <Card sx={{ mb: 3 }}>
      <CardContent>
        <Typography variant="h6" gutterBottom>
          📊 Map Statistics
        </Typography>
        <Box sx={{ display: 'flex', gap: 2, flexWrap: 'wrap' }}>
          <Chip
            icon={<LocationOn />}
            label={`${totalReports} Total Reports`}
            color="primary"
            variant="outlined"
          />
          <Chip
            icon={<CheckCircle />}
            label={`${lanternFlyReports} Lantern Flies`}
            color="success"
            variant="outlined"
          />
          <Chip
            icon={<Cancel />}
            label={`${notLanternFlyReports} Other Insects`}
            color="default"
            variant="outlined"
          />
        </Box>
      </CardContent>
    </Card>
  );
};

const MapComponent: React.FC<{ locations: MapLocation[] }> = ({ locations }) => {
  const map = useMap();

  useEffect(() => {
    if (locations.length > 0) {
      // Calculate bounds to fit all markers
      const bounds = locations.map(loc => [loc.latitude, loc.longitude] as [number, number]);
      map.fitBounds(bounds, { padding: [20, 20] });
    }
  }, [locations, map]);

  return null;
};

const MapPage: React.FC = () => {
  const [userLocation, setUserLocation] = useState<[number, number] | null>(null);
  const [mapError, setMapError] = useState<string | null>(null);

  const { data: locations = [], isLoading, error } = useQuery(
    'mapLocations',
    photoService.getMapLocations,
    {
      refetchInterval: 30000, // Refetch every 30 seconds
      onError: (err) => {
        console.error('Map data error:', err);
        setMapError('Failed to load map data');
      }
    }
  );

  useEffect(() => {
    // Get user's current location
    if (navigator.geolocation) {
      navigator.geolocation.getCurrentPosition(
        (position) => {
          setUserLocation([position.coords.latitude, position.coords.longitude]);
        },
        (error) => {
          console.warn('Geolocation error:', error);
        }
      );
    }
  }, []);

  if (isLoading) {
    return (
      <Container maxWidth="lg" sx={{ py: 4 }}>
        <Typography variant="h4" gutterBottom align="center">
          🗺️ Lantern Fly Map
        </Typography>
        <Box sx={{ display: 'flex', justifyContent: 'center', mt: 4 }}>
          <Typography>Loading map data...</Typography>
        </Box>
      </Container>
    );
  }

  if (error || mapError) {
    return (
      <Container maxWidth="lg" sx={{ py: 4 }}>
        <Typography variant="h4" gutterBottom align="center">
          🗺️ Lantern Fly Map
        </Typography>
        <Alert severity="error" sx={{ mt: 2 }}>
          {mapError || 'Failed to load map data. Please try again later.'}
        </Alert>
      </Container>
    );
  }

  return (
    <Container maxWidth="lg" sx={{ py: 4 }}>
      <Typography variant="h4" gutterBottom align="center">
        🗺️ Lantern Fly Map
      </Typography>
      <Typography variant="body1" color="text.secondary" align="center" sx={{ mb: 4 }}>
        Explore reported lantern fly locations in your area
      </Typography>

      <MapStats locations={locations} />

      <Card>
        <CardContent sx={{ p: 0 }}>
          <Box sx={{ height: '500px', width: '100%' }}>
            {typeof window !== 'undefined' && (
              <MapContainer
                center={userLocation || [40.7128, -74.0060]} // Default to NYC
                zoom={10}
                style={{ height: '100%', width: '100%' }}
              >
              <TileLayer
                attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
                url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
              />
              
              <MapComponent locations={locations} />
              
              {locations.map((location) => (
                <Marker
                  key={location.id}
                  position={[location.latitude, location.longitude]}
                  icon={location.is_lantern_fly ? lanternFlyIcon : notLanternFlyIcon}
                >
                  <Popup>
                    <Box>
                      <Typography variant="subtitle2" gutterBottom>
                        {location.is_lantern_fly ? (
                          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                            <BugReport color="success" />
                            Lantern Fly Detected
                          </Box>
                        ) : (
                          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                            <Cancel color="default" />
                            Not a Lantern Fly
                          </Box>
                        )}
                      </Typography>
                      <Typography variant="body2" color="text.secondary">
                        Confidence: {(location.confidence_score * 100).toFixed(1)}%
                      </Typography>
                      <Typography variant="body2" color="text.secondary">
                        Reported: {new Date(location.created_at).toLocaleDateString()}
                      </Typography>
                    </Box>
                  </Popup>
                </Marker>
              ))}
              </MapContainer>
            )}
            {typeof window === 'undefined' && (
              <Box sx={{ 
                height: '100%', 
                display: 'flex', 
                alignItems: 'center', 
                justifyContent: 'center',
                backgroundColor: 'grey.100'
              }}>
                <Typography>Loading map...</Typography>
              </Box>
            )}
          </Box>
        </CardContent>
      </Card>

      <Card sx={{ mt: 3, bgcolor: 'info.light', color: 'white' }}>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            🧭 Map Legend
          </Typography>
          <Box sx={{ display: 'flex', gap: 3, flexWrap: 'wrap' }}>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
              <Box sx={{ width: 20, height: 20, bgcolor: '#4CAF50', borderRadius: '50%' }} />
              <Typography variant="body2">Confirmed Lantern Fly</Typography>
            </Box>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
              <Box sx={{ width: 20, height: 20, bgcolor: '#757575', borderRadius: '50%' }} />
              <Typography variant="body2">Not a Lantern Fly</Typography>
            </Box>
          </Box>
        </CardContent>
      </Card>

      <VerifiedLanternfliesTable />
    </Container>
  );
};

// Component for the verified lanternflies table
const VerifiedLanternfliesTable: React.FC = () => {
  const [page, setPage] = useState(1);
  const limit = 10;

  const { data, isLoading, error } = useQuery(
    ['verified-lanternflies', page, limit],
    () => verifiedLanternfliesService.getAllSightings(page, limit),
    {
      keepPreviousData: true,
      refetchInterval: 30000, // Refetch every 30 seconds
    }
  );

  const handlePageChange = (event: React.ChangeEvent<unknown>, value: number) => {
    setPage(value);
  };

  if (isLoading) {
    return (
      <Card sx={{ mt: 3 }}>
        <CardContent>
          <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', py: 4 }}>
            <CircularProgress />
            <Typography variant="body1" sx={{ ml: 2 }}>
              Loading verified lanternfly sightings...
            </Typography>
          </Box>
        </CardContent>
      </Card>
    );
  }

  if (error) {
    return (
      <Card sx={{ mt: 3 }}>
        <CardContent>
          <Alert severity="error">
            Failed to load verified lanternfly sightings. Please try again later.
          </Alert>
        </CardContent>
      </Card>
    );
  }

  const sightings = data?.data?.sightings || [];
  const pagination = data?.data?.pagination;

  return (
    <Card sx={{ mt: 3 }}>
      <CardContent>
        <Typography variant="h5" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <BugReport color="primary" />
          Verified Lanternfly Sightings
        </Typography>
        
        {sightings.length === 0 ? (
          <Alert severity="info">
            No verified lanternfly sightings found. Start taking photos to see them here!
          </Alert>
        ) : (
          <>
            <TableContainer component={Paper} sx={{ mt: 2 }}>
              <Table>
                <TableHead>
                  <TableRow>
                    <TableCell>User</TableCell>
                    <TableCell>Location</TableCell>
                    <TableCell>Coordinates</TableCell>
                    <TableCell>Image</TableCell>
                    <TableCell>Confidence</TableCell>
                    <TableCell>Date/Time</TableCell>
                    <TableCell>Points</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {sightings.map((sighting) => (
                    <TableRow key={sighting.id} hover>
                      <TableCell>
                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                          <Avatar sx={{ width: 32, height: 32, bgcolor: 'primary.main' }}>
                            {sighting.username.charAt(0).toUpperCase()}
                          </Avatar>
                          <Typography variant="body2" fontWeight="medium">
                            {sighting.username}
                          </Typography>
                        </Box>
                      </TableCell>
                      <TableCell>
                        <Typography variant="body2">
                          {sighting.location_name || 'Unknown Location'}
                        </Typography>
                      </TableCell>
                      <TableCell>
                        <Typography variant="body2" fontFamily="monospace">
                          {sighting.latitude?.toFixed(4)}, {sighting.longitude?.toFixed(4)}
                        </Typography>
                      </TableCell>
                      <TableCell>
                        <Box
                          component="img"
                          src={`http://localhost:5000${sighting.image_url}`}
                          alt="Lanternfly sighting"
                          sx={{
                            width: 60,
                            height: 60,
                            objectFit: 'cover',
                            borderRadius: 1,
                            border: '1px solid',
                            borderColor: 'divider',
                            cursor: 'pointer',
                            '&:hover': {
                              opacity: 0.8,
                            }
                          }}
                          onClick={() => {
                            // Open image in new tab
                            window.open(`http://localhost:5000${sighting.image_url}`, '_blank');
                          }}
                        />
                      </TableCell>
                      <TableCell>
                        <Chip
                          label={`${(sighting.confidence_score * 100).toFixed(1)}%`}
                          color={sighting.confidence_score > 0.8 ? 'success' : 'warning'}
                          size="small"
                        />
                      </TableCell>
                      <TableCell>
                        <Typography variant="body2">
                          {new Date(sighting.sighting_date).toLocaleDateString()}
                        </Typography>
                        <Typography variant="caption" color="text.secondary">
                          {new Date(sighting.sighting_date).toLocaleTimeString()}
                        </Typography>
                      </TableCell>
                      <TableCell>
                        <Chip
                          label={`+${sighting.points_awarded}`}
                          color="primary"
                          size="small"
                        />
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </TableContainer>

            {pagination && pagination.totalPages > 1 && (
              <Box sx={{ display: 'flex', justifyContent: 'center', mt: 3 }}>
                <Pagination
                  count={pagination.totalPages}
                  page={page}
                  onChange={handlePageChange}
                  color="primary"
                  showFirstButton
                  showLastButton
                />
              </Box>
            )}

            <Typography variant="body2" color="text.secondary" sx={{ mt: 2, textAlign: 'center' }}>
              Showing {sightings.length} of {pagination?.total || 0} verified lanternfly sightings
            </Typography>
          </>
        )}
      </CardContent>
    </Card>
  );
};

export default MapPage;
