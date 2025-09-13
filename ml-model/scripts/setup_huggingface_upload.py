#!/usr/bin/env python3
"""
Helper script to set up Hugging Face authentication and upload the processed dataset.
"""

import subprocess
import sys
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_huggingface_auth():
    """Check if user is authenticated with Hugging Face."""
    try:
        from huggingface_hub import whoami
        user_info = whoami()
        logger.info(f"Already authenticated as: {user_info['name']}")
        return True
    except Exception:
        logger.info("Not authenticated with Hugging Face")
        return False

def authenticate_huggingface():
    """Guide user through Hugging Face authentication."""
    logger.info("Setting up Hugging Face authentication...")
    logger.info("You need to authenticate with Hugging Face to upload the dataset.")
    logger.info("Please follow these steps:")
    logger.info("1. Go to https://huggingface.co/settings/tokens")
    logger.info("2. Create a new token with 'write' permissions")
    logger.info("3. Copy the token")
    logger.info("4. Run: huggingface-cli login")
    logger.info("5. Paste your token when prompted")
    
    input("Press Enter when you have completed the authentication...")
    
    # Check if authentication was successful
    if check_huggingface_auth():
        logger.info("Authentication successful!")
        return True
    else:
        logger.error("Authentication failed. Please try again.")
        return False

def upload_dataset():
    """Upload the processed dataset."""
    try:
        from upload_processed_dataset import main as upload_main
        upload_main()
        return True
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        return False

def main():
    """Main setup function."""
    logger.info("Setting up Hugging Face dataset upload...")
    
    # Check if already authenticated
    if check_huggingface_auth():
        logger.info("Already authenticated. Proceeding with upload...")
    else:
        # Guide through authentication
        if not authenticate_huggingface():
            logger.error("Authentication failed. Cannot proceed with upload.")
            return
    
    # Upload the dataset
    logger.info("Starting dataset upload...")
    if upload_dataset():
        logger.info("Dataset upload completed successfully!")
        logger.info("You can now find your dataset at: https://huggingface.co/datasets/ddecosmo/lanternfly_swatter_training")
    else:
        logger.error("Dataset upload failed.")

if __name__ == "__main__":
    main()
