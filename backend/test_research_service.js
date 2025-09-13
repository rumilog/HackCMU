// Test the research dataset service directly
import { researchDatasetService } from './src/services/researchDatasetService.js';

console.log('🧪 Testing Research Dataset Service');
console.log('=' * 40);

try {
  console.log('✅ Service imported successfully');
  
  // Test dataset access
  console.log('🔍 Testing dataset access...');
  const hasAccess = await researchDatasetService.checkDatasetAccess();
  console.log(`Dataset access: ${hasAccess ? '✅ Yes' : '❌ No'}`);
  
  // Test dataset stats
  console.log('📊 Testing dataset stats...');
  const stats = await researchDatasetService.getDatasetStats();
  console.log('Dataset stats:', stats);
  
} catch (error) {
  console.error('❌ Service test failed:', error);
}
