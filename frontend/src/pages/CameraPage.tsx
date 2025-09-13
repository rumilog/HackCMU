import React, { useState, useRef, useCallback } from 'react';
import {
  Container,
  Paper,
  Typography,
  Button,
  Box,
  Alert,
  CircularProgress,
  Card,
  CardContent,
  Chip,
  LinearProgress,
} from '@mui/material';
import {
  CameraAlt,
  PhotoCamera,
  LocationOn,
  Upload,
  CheckCircle,
  Cancel,
} from '@mui/icons-material';
import { useQueryClient } from 'react-query';
import { photoService } from '../services/photoService';
import { ClassificationResult } from '../types';
import toast from 'react-hot-toast';

const CameraPage: React.FC = () => {
  const [capturedImage, setCapturedImage] = useState<string | null>(null);
  const [isCapturing, setIsCapturing] = useState(false);
  const [isClassifying, setIsClassifying] = useState(false);
  const [classificationResult, setClassificationResult] = useState<ClassificationResult | null>(null);
  const [location, setLocation] = useState<{ latitude: number; longitude: number } | null>(null);
  const [locationError, setLocationError] = useState<string | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  const [isVerifying, setIsVerifying] = useState(false);
  const [verificationResult, setVerificationResult] = useState<boolean | null>(null);

  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const queryClient = useQueryClient();

  const startCamera = useCallback(async () => {
    try {
      setIsCapturing(true);
      setLocationError(null);

      // Get user's location
      if (navigator.geolocation) {
        navigator.geolocation.getCurrentPosition(
          (position) => {
            setLocation({
              latitude: position.coords.latitude,
              longitude: position.coords.longitude,
            });
          },
          (error) => {
            setLocationError('Location access denied. Photo will be uploaded without location data.');
            console.warn('Geolocation error:', error);
          }
        );
      } else {
        setLocationError('Geolocation not supported by this browser.');
      }

      // Check if getUserMedia is supported
      if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        throw new Error('Camera not supported on this device');
      }

      // Start camera
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { 
          facingMode: 'environment', // Use back camera on mobile
          width: { ideal: 1280 },
          height: { ideal: 720 }
        },
      });
      
      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
      }
    } catch (error: any) {
      console.error('Camera error:', error);
      let errorMessage = 'Failed to access camera. Please check permissions.';
      
      if (error.name === 'NotAllowedError') {
        errorMessage = 'Camera access denied. Please allow camera permissions and try again.';
      } else if (error.name === 'NotFoundError') {
        errorMessage = 'No camera found on this device.';
      } else if (error.name === 'NotSupportedError') {
        errorMessage = 'Camera not supported on this device.';
      }
      
      toast.error(errorMessage);
      setIsCapturing(false);
    }
  }, []);

  const stopCamera = useCallback(() => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop());
      streamRef.current = null;
    }
    setIsCapturing(false);
  }, []);

  const capturePhoto = useCallback(() => {
    if (!videoRef.current || !canvasRef.current) return;

    const video = videoRef.current;
    const canvas = canvasRef.current;
    const context = canvas.getContext('2d');

    if (!context) return;

    // Set canvas size to match video
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    // Draw video frame to canvas
    context.drawImage(video, 0, 0);

    // Convert to blob
    canvas.toBlob((blob) => {
      if (blob) {
        console.log('Canvas blob created:', {
          size: blob.size,
          type: blob.type
        });
        
        // Ensure the blob has the correct MIME type
        const typedBlob = new Blob([blob], { type: 'image/jpeg' });
        console.log('Typed canvas blob:', {
          size: typedBlob.size,
          type: typedBlob.type
        });
        
        const imageUrl = URL.createObjectURL(typedBlob);
        setCapturedImage(imageUrl);
        stopCamera();
      }
    }, 'image/jpeg', 0.8);
  }, [stopCamera]);

  const classifyImage = async () => {
    if (!capturedImage) return;

    try {
      setIsClassifying(true);
      
      // Convert image URL to File
      const response = await fetch(capturedImage);
      const blob = await response.blob();
      
      console.log('Fetched blob:', {
        size: blob.size,
        type: blob.type
      });
      
      // Ensure the blob has the correct MIME type
      const typedBlob = new Blob([blob], { type: 'image/jpeg' });
      const file = new File([typedBlob], 'photo.jpg', { type: 'image/jpeg' });
      
      console.log('Created file from typed blob:', {
        name: file.name,
        size: file.size,
        type: file.type
      });

      console.log('Starting classification...');
      console.log('File object:', {
        name: file.name,
        size: file.size,
        type: file.type,
        lastModified: file.lastModified
      });
      
      // Test if the file is actually readable
      const reader = new FileReader();
      reader.onload = (e) => {
        console.log('File reader result:', e.target?.result ? 'File is readable' : 'File is not readable');
      };
      reader.readAsDataURL(file);
      console.log('Original blob:', {
        size: blob.size,
        type: blob.type
      });
      console.log('Typed blob:', {
        size: typedBlob.size,
        type: typedBlob.type
      });
      
      const result = await photoService.classifyPhoto(file);
      console.log('Classification result:', result);
      
      // Validate the result structure
      if (!result || typeof result.is_lantern_fly !== 'boolean') {
        throw new Error('Invalid classification result received');
      }
      
      setClassificationResult(result);
      toast.success('Image classified successfully!');
    } catch (error: any) {
      console.error('Classification error:', error);
      console.error('Error details:', {
        message: error.message,
        stack: error.stack,
        response: error.response
      });
      toast.error(error.message || 'Classification failed');
    } finally {
      setIsClassifying(false);
    }
  };

  const handleVerification = async (isLanternFly: boolean) => {
    if (!capturedImage || !classificationResult) return;

    try {
      setIsVerifying(true);
      setVerificationResult(isLanternFly);
      
      // If user confirms it's a lanternfly, proceed with upload
      if (isLanternFly) {
        await uploadPhoto();
      } else {
        // If user says it's not a lanternfly, show message and reset
        toast.info('Classification corrected. No points awarded.');
        setCapturedImage(null);
        setClassificationResult(null);
        setLocation(null);
        setLocationError(null);
        setVerificationResult(null);
      }
      
    } catch (error: any) {
      console.error('Verification error:', error);
      toast.error(error.message || 'Verification failed');
    } finally {
      setIsVerifying(false);
    }
  };

  const uploadPhoto = async () => {
    if (!capturedImage || !classificationResult) return;

    try {
      setIsUploading(true);
      
      // Convert image URL to File
      const response = await fetch(capturedImage);
      const blob = await response.blob();
      const file = new File([blob], 'photo.jpg', { type: 'image/jpeg' });

      const photoData = {
        file,
        latitude: location?.latitude,
        longitude: location?.longitude,
      };

      await photoService.uploadPhoto(photoData);
      
      // Invalidate queries to refresh data
      queryClient.invalidateQueries('userStats');
      queryClient.invalidateQueries('userPhotos');
      queryClient.invalidateQueries('mapLocations');
      
      toast.success('Photo uploaded successfully!');
      
      // Reset state
      setCapturedImage(null);
      setClassificationResult(null);
      setLocation(null);
      setLocationError(null);
      setVerificationResult(null);
    } catch (error: any) {
      console.error('Upload error:', error);
      toast.error(error.message || 'Upload failed');
    } finally {
      setIsUploading(false);
    }
  };

  const resetCapture = () => {
    setCapturedImage(null);
    setClassificationResult(null);
    setLocation(null);
    setLocationError(null);
    setVerificationResult(null);
  };

  return (
    <Container maxWidth="md" sx={{ py: 4 }}>
      <Typography variant="h4" gutterBottom align="center">
        📸 Capture Lantern Fly
      </Typography>
      <Typography variant="body1" color="text.secondary" align="center" sx={{ mb: 4 }}>
        Take a photo to identify and track dead lantern flies
      </Typography>

      {!capturedImage ? (
        <Paper elevation={3} sx={{ p: 3 }}>
          {!isCapturing ? (
            <Box sx={{ textAlign: 'center' }}>
              <CameraAlt sx={{ fontSize: 80, color: 'primary.main', mb: 2 }} />
              <Typography variant="h6" gutterBottom>
                Ready to take a photo?
              </Typography>
              <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
                Make sure you have good lighting and the lantern fly is clearly visible
              </Typography>
              <Button
                variant="contained"
                size="large"
                startIcon={<PhotoCamera />}
                onClick={startCamera}
                sx={{ py: 1.5, px: 4 }}
              >
                Start Camera
              </Button>
            </Box>
          ) : (
            <Box>
              <Box sx={{ position: 'relative', mb: 2 }}>
                <video
                  ref={videoRef}
                  autoPlay
                  playsInline
                  muted
                  style={{
                    width: '100%',
                    height: 'auto',
                    borderRadius: 8,
                  }}
                />
                <canvas ref={canvasRef} style={{ display: 'none' }} />
              </Box>
              
              {locationError && (
                <Alert severity="warning" sx={{ mb: 2 }}>
                  {locationError}
                </Alert>
              )}
              
              {location && (
                <Alert severity="success" sx={{ mb: 2 }}>
                  <LocationOn sx={{ mr: 1 }} />
                  Location captured: {location.latitude.toFixed(6)}, {location.longitude.toFixed(6)}
                </Alert>
              )}

              <Box sx={{ display: 'flex', gap: 2, justifyContent: 'center' }}>
                <Button
                  variant="contained"
                  size="large"
                  startIcon={<CameraAlt />}
                  onClick={capturePhoto}
                  sx={{ py: 1.5, px: 4 }}
                >
                  Capture Photo
                </Button>
                <Button
                  variant="outlined"
                  size="large"
                  onClick={stopCamera}
                  sx={{ py: 1.5, px: 4 }}
                >
                  Cancel
                </Button>
              </Box>
            </Box>
          )}
        </Paper>
      ) : (
        <Box>
          {/* Captured Image */}
          <Card sx={{ mb: 3 }}>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Captured Photo
              </Typography>
              <Box sx={{ textAlign: 'center' }}>
                <img
                  src={capturedImage}
                  alt="Captured lantern fly"
                  style={{
                    maxWidth: '100%',
                    height: 'auto',
                    borderRadius: 8,
                    maxHeight: '400px',
                  }}
                />
              </Box>
            </CardContent>
          </Card>

          {/* Classification */}
          {!classificationResult ? (
            <Card sx={{ mb: 3 }}>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Classify Image
                </Typography>
                <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                  Our AI will analyze the image to determine if it contains a dead lantern fly
                </Typography>
                <Button
                  variant="contained"
                  size="large"
                  onClick={classifyImage}
                  disabled={isClassifying}
                  startIcon={isClassifying ? <CircularProgress size={20} /> : <Upload />}
                  sx={{ py: 1.5, px: 4 }}
                >
                  {isClassifying ? 'Classifying...' : 'Classify Image'}
                </Button>
              </CardContent>
            </Card>
          ) : (
            <Card sx={{ mb: 3 }}>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Classification Result
                </Typography>
                
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, mb: 2 }}>
                  {classificationResult.is_lantern_fly ? (
                    <Chip
                      icon={<CheckCircle />}
                      label="Lantern Fly Detected"
                      color="success"
                      variant="filled"
                    />
                  ) : (
                    <Chip
                      icon={<Cancel />}
                      label="Not a Lantern Fly"
                      color="default"
                      variant="filled"
                    />
                  )}
                  <Typography variant="body2" color="text.secondary">
                    Confidence: {((classificationResult.confidence_score || 0) * 100).toFixed(1)}%
                  </Typography>
                </Box>

                <LinearProgress
                  variant="determinate"
                  value={(classificationResult.confidence_score || 0) * 100}
                  color={classificationResult.is_lantern_fly ? 'primary' : 'secondary'}
                  sx={{ mb: 2 }}
                />

                <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                  Points to be awarded: {classificationResult.points_awarded || 0}
                </Typography>

                {/* Verification status */}
                {verificationResult !== null && (
                  <Box sx={{ mb: 2, p: 2, borderRadius: 2, bgcolor: verificationResult ? 'success.light' : 'error.light' }}>
                    <Typography variant="body2" color={verificationResult ? 'success.contrastText' : 'error.contrastText'} align="center">
                      {verificationResult ? '✅ Verified as lanternfly - proceeding with upload' : '❌ Not a lanternfly - no points awarded'}
                    </Typography>
                  </Box>
                )}

                {/* Verification buttons for lanternfly detection */}
                {classificationResult.is_lantern_fly && verificationResult === null ? (
                  <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
                    <Typography variant="h6" color="primary" align="center" sx={{ mb: 1 }}>
                      🤔 Is this really a lanternfly?
                    </Typography>
                    <Typography variant="body2" color="text.secondary" align="center" sx={{ mb: 2 }}>
                      Please verify the AI classification to ensure accuracy
                    </Typography>
                    <Box sx={{ display: 'flex', gap: 2, justifyContent: 'center' }}>
                      <Button
                        variant="contained"
                        color="success"
                        size="large"
                        onClick={() => handleVerification(true)}
                        disabled={isVerifying}
                        startIcon={isVerifying ? <CircularProgress size={20} /> : <CheckCircle />}
                        sx={{ py: 1.5, px: 4 }}
                      >
                        {isVerifying ? 'Verifying...' : 'Yes, it\'s a lanternfly'}
                      </Button>
                      <Button
                        variant="contained"
                        color="error"
                        size="large"
                        onClick={() => handleVerification(false)}
                        disabled={isVerifying}
                        startIcon={isVerifying ? <CircularProgress size={20} /> : <Cancel />}
                        sx={{ py: 1.5, px: 4 }}
                      >
                        {isVerifying ? 'Verifying...' : 'No, it\'s not'}
                      </Button>
                    </Box>
                  </Box>
                ) : (
                  /* Regular upload buttons for non-lanternfly or after verification */
                  <Box sx={{ display: 'flex', gap: 2 }}>
                    <Button
                      variant="contained"
                      size="large"
                      onClick={uploadPhoto}
                      disabled={isUploading}
                      startIcon={isUploading ? <CircularProgress size={20} /> : <Upload />}
                      sx={{ py: 1.5, px: 4 }}
                    >
                      {isUploading ? 'Uploading...' : 'Upload Photo'}
                    </Button>
                    <Button
                      variant="outlined"
                      size="large"
                      onClick={resetCapture}
                      sx={{ py: 1.5, px: 4 }}
                    >
                      Take Another
                    </Button>
                  </Box>
                )}
              </CardContent>
            </Card>
          )}
        </Box>
      )}
    </Container>
  );
};

export default CameraPage;
