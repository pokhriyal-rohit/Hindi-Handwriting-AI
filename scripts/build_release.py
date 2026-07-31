import os
import zipfile
import shutil
from pathlib import Path

def build_release():
    release_dir = "releases"
    os.makedirs(release_dir, exist_ok=True)
    
    zip_name = "v1.1.0-colab-ready.zip"
    zip_path = os.path.join(release_dir, zip_name)
    
    if os.path.exists(zip_path):
        os.remove(zip_path)
        
    print(f"Building release archive: {zip_path}")
    
    # Files and folders to include
    includes = [
        "src",
        "configs",
        "scripts",
        "data/canonical",
        "main.py",
        "requirements.txt",
        "requirements_colab.txt",
        "Colab_Training.ipynb",
        "VERSION",
        "CHANGELOG.md",
        "LICENSE",
        "README.md",
        "ARCHIVED_DATASETS.md",
        "COLAB_SETUP.md"
    ]
    
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for item in includes:
            if not os.path.exists(item):
                print(f"Warning: {item} not found, skipping.")
                continue
                
            if os.path.isfile(item):
                zipf.write(item, item)
            elif os.path.isdir(item):
                for root, _, files in os.walk(item):
                    # Skip pycache and hidden dirs
                    if "__pycache__" in root or "/." in root.replace("\\", "/"):
                        continue
                    for file in files:
                        file_path = os.path.join(root, file)
                        zipf.write(file_path, file_path)
                        
    print("Release package successfully built.")

if __name__ == "__main__":
    build_release()
