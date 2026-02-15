#!/usr/bin/env python3
"""
Data Download Script
Downloads UCI Diabetes dataset for Hospital Readmission project
"""

import urllib.request
import zipfile
import os

def download_dataset():
    """Download and extract the UCI Diabetes dataset"""
    
    url = "https://archive.ics.uci.edu/static/public/296/diabetes+130-us+hospitals+for+years+1999-2008.zip"
    zip_path = "diabetes_dataset.zip"
    
    print("📥 Downloading dataset...")
    print(f"URL: {url}")
    
    # Download with progress
    def report_progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        percent = min(downloaded * 100 / total_size, 100)
        print(f"\rProgress: {percent:.1f}%", end="", flush=True)
    
    try:
        urllib.request.urlretrieve(url, zip_path, reporthook=report_progress)
        print("\n✅ Download complete!")
        
        # Extract
        print("📦 Extracting files...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(".")
        print("✅ Extraction complete!")
        
        # Clean up zip
        os.remove(zip_path)
        print("🧹 Cleaned up zip file")
        
        # List files
        print("\n📁 Downloaded files:")
        for f in os.listdir("."):
            if f.endswith('.csv'):
                size = os.path.getsize(f) / (1024*1024)  # MB
                print(f"  - {f} ({size:.2f} MB)")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return False

if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    success = download_dataset()
    exit(0 if success else 1)
