// Image utility functions for the Lantern Fly Tracker

export interface ImageDimensions {
  width: number;
  height: number;
}

export const compressImage = (
  file: File,
  maxWidth: number = 1920,
  maxHeight: number = 1080,
  quality: number = 0.8
): Promise<File> => {
  return new Promise((resolve, reject) => {
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    const img = new Image();

    img.onload = () => {
      // Calculate new dimensions
      let { width, height } = img;
      
      if (width > height) {
        if (width > maxWidth) {
          height = (height * maxWidth) / width;
          width = maxWidth;
        }
      } else {
        if (height > maxHeight) {
          width = (width * maxHeight) / height;
          height = maxHeight;
        }
      }

      // Set canvas dimensions
      canvas.width = width;
      canvas.height = height;

      // Draw and compress
      ctx?.drawImage(img, 0, 0, width, height);
      
      canvas.toBlob(
        (blob) => {
          if (blob) {
            const compressedFile = new File([blob], file.name, {
              type: 'image/jpeg',
              lastModified: Date.now(),
            });
            resolve(compressedFile);
          } else {
            reject(new Error('Failed to compress image'));
          }
        },
        'image/jpeg',
        quality
      );
    };

    img.onerror = () => {
      reject(new Error('Failed to load image'));
    };

    img.src = URL.createObjectURL(file);
  });
};

export const getImageDimensions = (file: File): Promise<ImageDimensions> => {
  return new Promise((resolve, reject) => {
    const img = new Image();
    
    img.onload = () => {
      resolve({
        width: img.naturalWidth,
        height: img.naturalHeight,
      });
    };

    img.onerror = () => {
      reject(new Error('Failed to load image'));
    };

    img.src = URL.createObjectURL(file);
  });
};

export const createThumbnail = (
  file: File,
  size: number = 200
): Promise<string> => {
  return new Promise((resolve, reject) => {
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    const img = new Image();

    canvas.width = size;
    canvas.height = size;

    img.onload = () => {
      // Calculate thumbnail dimensions (maintain aspect ratio)
      const aspectRatio = img.width / img.height;
      let thumbWidth = size;
      let thumbHeight = size;

      if (aspectRatio > 1) {
        thumbHeight = size / aspectRatio;
      } else {
        thumbWidth = size * aspectRatio;
      }

      // Center the thumbnail
      const x = (size - thumbWidth) / 2;
      const y = (size - thumbHeight) / 2;

      // Draw thumbnail
      ctx?.drawImage(img, x, y, thumbWidth, thumbHeight);
      
      const thumbnailUrl = canvas.toDataURL('image/jpeg', 0.8);
      resolve(thumbnailUrl);
    };

    img.onerror = () => {
      reject(new Error('Failed to create thumbnail'));
    };

    img.src = URL.createObjectURL(file);
  });
};

export const validateImageFile = (file: File): { valid: boolean; error?: string } => {
  // Check file type
  if (!file.type.startsWith('image/')) {
    return { valid: false, error: 'File must be an image' };
  }

  // Check file size (max 10MB)
  const maxSize = 10 * 1024 * 1024; // 10MB
  if (file.size > maxSize) {
    return { valid: false, error: 'Image must be smaller than 10MB' };
  }

  // Check supported formats
  const supportedFormats = ['image/jpeg', 'image/jpg', 'image/png', 'image/webp'];
  if (!supportedFormats.includes(file.type)) {
    return { valid: false, error: 'Unsupported image format. Please use JPEG, PNG, or WebP' };
  }

  return { valid: true };
};

export const formatFileSize = (bytes: number): string => {
  if (bytes === 0) return '0 Bytes';

  const k = 1024;
  const sizes = ['Bytes', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));

  return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
};

export const getImageOrientation = (file: File): Promise<number> => {
  return new Promise((resolve) => {
    const reader = new FileReader();
    
    reader.onload = (e) => {
      const arrayBuffer = e.target?.result as ArrayBuffer;
      const dataView = new DataView(arrayBuffer);
      
      // Check for EXIF data
      if (dataView.getUint16(0) !== 0xFFD8) {
        resolve(1); // Default orientation
        return;
      }

      let offset = 2;
      let marker;

      while (offset < dataView.byteLength) {
        marker = dataView.getUint16(offset);
        
        if (marker === 0xFFE1) {
          // Found EXIF data
          const exifLength = dataView.getUint16(offset + 2);
          
          if (dataView.getUint32(offset + 4) === 0x45786966) {
            // Found EXIF header
            const tiffOffset = offset + 10;
            const littleEndian = dataView.getUint16(tiffOffset) === 0x4949;
            
            // Read orientation tag
            const ifdOffset = tiffOffset + dataView.getUint32(tiffOffset + 4, littleEndian);
            const numEntries = dataView.getUint16(ifdOffset, littleEndian);
            
            for (let i = 0; i < numEntries; i++) {
              const entryOffset = ifdOffset + 2 + (i * 12);
              const tag = dataView.getUint16(entryOffset, littleEndian);
              
              if (tag === 0x0112) { // Orientation tag
                resolve(dataView.getUint16(entryOffset + 8, littleEndian));
                return;
              }
            }
          }
          break;
        }
        
        offset += 2 + dataView.getUint16(offset + 2);
      }
      
      resolve(1); // Default orientation
    };
    
    reader.readAsArrayBuffer(file.slice(0, 64 * 1024)); // Read first 64KB
  });
};
