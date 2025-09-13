// Test script to simulate frontend file creation
const fs = require('fs');
const path = require('path');

// Simulate what the frontend does
function testFileCreation() {
  console.log('Testing file creation simulation...');
  
  // Read a sample image
  const sampleImage = 'ml-model/data/processed/train/lantern_fly/lantern_fly_0001.jpg';
  
  if (!fs.existsSync(sampleImage)) {
    console.log('Sample image not found');
    return;
  }
  
  const imageBuffer = fs.readFileSync(sampleImage);
  console.log('Image buffer size:', imageBuffer.length);
  console.log('Image buffer type:', typeof imageBuffer);
  
  // Simulate creating a File-like object
  const fileLike = {
    name: 'photo.jpg',
    size: imageBuffer.length,
    type: 'image/jpeg',
    buffer: imageBuffer
  };
  
  console.log('File-like object:', fileLike);
  
  // Test with requests
  const FormData = require('form-data');
  const formData = new FormData();
  formData.append('image', imageBuffer, {
    filename: 'photo.jpg',
    contentType: 'image/jpeg'
  });
  
  console.log('FormData created successfully');
  console.log('FormData headers:', formData.getHeaders());
}

testFileCreation();
