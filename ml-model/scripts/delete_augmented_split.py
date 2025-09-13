#!/usr/bin/env python3
"""
Script to delete the augmented split from Hugging Face dataset.
"""

import logging
from huggingface_hub import HfApi, whoami

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def delete_augmented_split(repo_name: str = "rlogh/lanternfly_swatter_training"):
    """Delete the augmented split from the Hugging Face dataset."""
    logger.info(f"Deleting augmented split from {repo_name}...")
    
    try:
        # Check authentication
        user_info = whoami()
        logger.info(f"Logged in as: {user_info['name']}")
        
        # Initialize API
        api = HfApi()
        
        # Delete the augmented split files
        try:
            # List files in the repo
            repo_files = api.list_repo_files(repo_name, repo_type="dataset")
            logger.info(f"Found files in repo: {repo_files}")
            
            # Delete augmented split files one by one
            augmented_files = [f for f in repo_files if "augmented" in f]
            if augmented_files:
                logger.info(f"Deleting augmented files: {augmented_files}")
                for file in augmented_files:
                    try:
                        api.delete_file(file, repo_id=repo_name, repo_type="dataset")
                        logger.info(f"Deleted {file}")
                    except Exception as e:
                        logger.warning(f"Could not delete {file}: {e}")
                logger.info("Augmented split deletion attempted!")
            else:
                logger.info("No augmented files found to delete")
                
        except Exception as e:
            logger.warning(f"Could not delete files directly: {e}")
            logger.info("You may need to manually delete the augmented split from the Hugging Face web interface")
        
    except Exception as e:
        logger.error(f"Failed to delete augmented split: {e}")
        raise

def main():
    """Main function."""
    try:
        delete_augmented_split()
        logger.info("Cleanup completed!")
    except Exception as e:
        logger.error(f"Cleanup failed: {e}")

if __name__ == "__main__":
    main()
