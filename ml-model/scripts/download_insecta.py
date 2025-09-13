#!/usr/bin/env python3
"""
Download and Organize Insecta Dataset
Downloads the Insecta dataset from Hugging Face and organizes it for training
"""

import os
import shutil
import logging
from pathlib import Path
from typing import List, Dict
import requests
import zipfile
from tqdm import tqdm
import json

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class InsectaDownloader:
    """Handles downloading and organizing the Insecta dataset"""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / "raw"
        self.insecta_dir = self.raw_dir / "insecta"
        
        # Create directories
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.insecta_dir.mkdir(parents=True, exist_ok=True)
        
        # Dataset information
        self.dataset_info = {
            "name": "Insecta",
            "source": "Hugging Face",
            "repository": "Genius-Society/insecta",
            "description": "Large-scale insect dataset for classification",
            "classes": "Multiple insect species (excluding lantern flies)",
            "size": "~10,000+ images",
            "format": "Images with class labels"
        }
    
    def download_file(self, url: str, destination: Path, description: str = "Downloading"):
        """Download a file with progress bar"""
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            
            with open(destination, 'wb') as f:
                with tqdm(total=total_size, unit='B', unit_scale=True, desc=description) as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))
            
            logger.info(f"Downloaded: {destination}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to download {url}: {e}")
            return False
    
    def download_insecta_dataset(self, method: str = "huggingface"):
        """Download Insecta dataset using specified method"""
        logger.info(f"Downloading Insecta dataset using method: {method}")
        
        if method == "huggingface":
            return self._download_from_huggingface()
        elif method == "manual":
            return self._provide_manual_instructions()
        else:
            logger.error(f"Unknown download method: {method}")
            return False
    
    def _download_from_huggingface(self):
        """Download from Hugging Face Hub"""
        try:
            from huggingface_hub import snapshot_download
            
            logger.info("Downloading Insecta dataset from Hugging Face...")
            
            # Download dataset
            dataset_path = snapshot_download(
                repo_id="Genius-Society/insecta",
                repo_type="dataset",
                local_dir=self.insecta_dir,
                local_dir_use_symlinks=False
            )
            
            logger.info(f"Dataset downloaded to: {dataset_path}")
            return True
            
        except ImportError:
            logger.error("huggingface_hub not installed. Install with: pip install huggingface_hub")
            return False
        except Exception as e:
            logger.error(f"Failed to download from Hugging Face: {e}")
            return False
    
    def _provide_manual_instructions(self):
        """Provide manual download instructions"""
        instructions = """
        MANUAL DOWNLOAD INSTRUCTIONS:
        
        1. Go to: https://huggingface.co/datasets/Genius-Society/insecta
        
        2. Click "Files and versions" tab
        
        3. Download the dataset files (usually a zip file)
        
        4. Extract the downloaded file to: data/raw/insecta/
        
        5. Ensure the structure looks like:
           data/raw/insecta/
           ├── train/
           │   ├── class1/
           │   ├── class2/
           │   └── ...
           └── test/
               ├── class1/
               ├── class2/
               └── ...
        
        6. Run this script again to organize the data
        
        Alternative: Use the Hugging Face CLI:
        pip install huggingface_hub
        huggingface-cli download Genius-Society/insecta --local-dir data/raw/insecta
        """
        
        print(instructions)
        
        # Save instructions to file
        with open("insecta_download_instructions.txt", 'w') as f:
            f.write(instructions)
        
        logger.info("Manual download instructions saved to: insecta_download_instructions.txt")
        return False
    
    def organize_insecta_data(self):
        """Organize downloaded Insecta data"""
        logger.info("Organizing Insecta dataset...")
        
        if not self.insecta_dir.exists():
            logger.error(f"Insecta directory not found: {self.insecta_dir}")
            return False
        
        # Find all image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        all_images = []
        
        for root, dirs, files in os.walk(self.insecta_dir):
            for file in files:
                if Path(file).suffix.lower() in image_extensions:
                    all_images.append(Path(root) / file)
        
        logger.info(f"Found {len(all_images)} images in Insecta dataset")
        
        if len(all_images) == 0:
            logger.error("No images found in Insecta dataset")
            return False
        
        # Create organized structure
        organized_dir = self.raw_dir / "insecta_organized"
        organized_dir.mkdir(exist_ok=True)
        
        # Copy all images to organized directory
        for i, image_path in enumerate(all_images):
            # Create new filename
            new_filename = f"insecta_{i:06d}{image_path.suffix}"
            new_path = organized_dir / new_filename
            
            # Copy image
            shutil.copy2(image_path, new_path)
        
        logger.info(f"Organized {len(all_images)} images to: {organized_dir}")
        
        # Create metadata file
        metadata = {
            "dataset": "Insecta",
            "source": "Genius-Society/insecta",
            "total_images": len(all_images),
            "organized_path": str(organized_dir),
            "original_path": str(self.insecta_dir),
            "image_extensions": list(image_extensions)
        }
        
        metadata_path = organized_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Metadata saved to: {metadata_path}")
        return True
    
    def sample_insecta_images(self, target_count: int = 2000, output_dir: str = None):
        """Sample images from Insecta dataset for training"""
        logger.info(f"Sampling {target_count} images from Insecta dataset...")
        
        # Find organized images
        organized_dir = self.raw_dir / "insecta_organized"
        if not organized_dir.exists():
            logger.error("Organized Insecta data not found. Run organize_insecta_data() first.")
            return False
        
        # Get all image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        all_images = []
        
        for file_path in organized_dir.glob("*"):
            if file_path.is_file() and file_path.suffix.lower() in image_extensions:
                all_images.append(file_path)
        
        if len(all_images) == 0:
            logger.error("No images found in organized Insecta data")
            return False
        
        # Sample images
        import random
        random.seed(42)  # For reproducibility
        
        if len(all_images) <= target_count:
            sampled_images = all_images
            logger.info(f"Using all {len(all_images)} images (less than target)")
        else:
            sampled_images = random.sample(all_images, target_count)
            logger.info(f"Sampled {len(sampled_images)} images from {len(all_images)} total")
        
        # Copy to output directory
        if output_dir is None:
            output_dir = self.data_dir / "processed" / "train" / "non_lantern_fly"
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        for i, image_path in enumerate(sampled_images):
            new_filename = f"insecta_{i:06d}{image_path.suffix}"
            new_path = output_path / new_filename
            shutil.copy2(image_path, new_path)
        
        logger.info(f"Copied {len(sampled_images)} images to: {output_path}")
        
        # Create sampling metadata
        sampling_metadata = {
            "source_dataset": "Insecta",
            "total_available": len(all_images),
            "sampled_count": len(sampled_images),
            "target_count": target_count,
            "output_directory": str(output_path),
            "sampling_method": "random",
            "random_seed": 42
        }
        
        metadata_path = output_path / "sampling_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(sampling_metadata, f, indent=2)
        
        logger.info(f"Sampling metadata saved to: {metadata_path}")
        return True
    
    def get_dataset_info(self):
        """Get information about the dataset"""
        info = self.dataset_info.copy()
        
        # Add local information
        if self.insecta_dir.exists():
            info["local_path"] = str(self.insecta_dir)
            info["exists_locally"] = True
            
            # Count images
            image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
            image_count = 0
            
            for root, dirs, files in os.walk(self.insecta_dir):
                for file in files:
                    if Path(file).suffix.lower() in image_extensions:
                        image_count += 1
            
            info["local_image_count"] = image_count
        else:
            info["exists_locally"] = False
        
        return info
    
    def print_dataset_info(self):
        """Print dataset information"""
        info = self.get_dataset_info()
        
        print("=" * 60)
        print(f"📊 {info['name']} Dataset Information")
        print("=" * 60)
        print(f"Source: {info['source']}")
        print(f"Repository: {info['repository']}")
        print(f"Description: {info['description']}")
        print(f"Classes: {info['classes']}")
        print(f"Size: {info['size']}")
        print(f"Format: {info['format']}")
        print()
        
        if info.get("exists_locally"):
            print(f"✅ Local Path: {info['local_path']}")
            print(f"📸 Local Images: {info['local_image_count']}")
        else:
            print("❌ Dataset not downloaded locally")
        
        print("=" * 60)

def main():
    """Main function for downloading and organizing Insecta dataset"""
    print("🦟 Insecta Dataset Downloader")
    print("=" * 40)
    
    # Initialize downloader
    downloader = InsectaDownloader()
    
    # Print dataset info
    downloader.print_dataset_info()
    
    # Check if dataset already exists
    if downloader.insecta_dir.exists():
        print(f"\n✅ Dataset already exists at: {downloader.insecta_dir}")
        
        # Ask if user wants to re-download
        response = input("\nDo you want to re-download? (y/N): ").strip().lower()
        if response != 'y':
            print("Skipping download. Organizing existing data...")
        else:
            # Re-download
            success = downloader.download_insecta_dataset("huggingface")
            if not success:
                print("Hugging Face download failed. Trying manual method...")
                downloader.download_insecta_dataset("manual")
                return
    else:
        # Download dataset
        print("\n📥 Downloading dataset...")
        success = downloader.download_insecta_dataset("huggingface")
        
        if not success:
            print("Hugging Face download failed. Trying manual method...")
            downloader.download_insecta_dataset("manual")
            return
    
    # Organize data
    print("\n📁 Organizing data...")
    success = downloader.organize_insecta_data()
    
    if not success:
        print("❌ Failed to organize data")
        return
    
    # Sample images for training
    print("\n🎯 Sampling images for training...")
    success = downloader.sample_insecta_images(target_count=2000)
    
    if success:
        print("\n✅ Insecta dataset setup completed successfully!")
        print("\n📋 Next steps:")
        print("1. Collect 500 lantern fly photos")
        print("2. Run data processing script")
        print("3. Train the model")
    else:
        print("❌ Failed to sample images")

if __name__ == "__main__":
    main()
