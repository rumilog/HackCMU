// Test script to simulate exactly what the frontend does
const fs = require('fs');
const FormData = require('form-data');
const axios = require('axios');

async function testFrontendSimulation() {
  console.log('🧪 Testing frontend simulation...');
  
  // Read a sample image
  const sampleImage = 'ml-model/data/processed/train/lantern_fly/lantern_fly_0001.jpg';
  
  if (!fs.existsSync(sampleImage)) {
    console.log('❌ Sample image not found');
    return;
  }
  
  const imageBuffer = fs.readFileSync(sampleImage);
  console.log('📸 Image buffer size:', imageBuffer.length);
  
  // Simulate what the frontend does
  const formData = new FormData();
  formData.append('image', imageBuffer, {
    filename: 'photo.jpg',
    contentType: 'image/jpeg'
  });
  
  console.log('📤 FormData created');
  console.log('📤 FormData headers:', formData.getHeaders());
  
  try {
    const response = await axios.post('http://localhost:5000/api/photos/classify', formData, {
      headers: {
        ...formData.getHeaders(),
      },
      timeout: 30000
    });
    
    console.log('✅ Response status:', response.status);
    console.log('✅ Response data:', response.data);
    
  } catch (error) {
    console.error('❌ Error:', error.message);
    if (error.response) {
      console.error('❌ Response status:', error.response.status);
      console.error('❌ Response data:', error.response.data);
    }
  }
}

testFrontendSimulation();
