"""
Setup script for backend dependencies
"""

import subprocess
import sys
import os

def install_requirements():
    """Install required packages"""
    try:
        print("Installing backend requirements...")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ])
        print("✅ Backend requirements installed successfully!")
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing requirements: {e}")
        return False
    
    return True

def setup_directories():
    """Create necessary directories"""
    directories = [
        "logs",
        "uploads",
        "static"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✅ Created directory: {directory}")

def main():
    """Main setup function"""
    print("🚀 Setting up FastAPI Backend...")
    
    # Install requirements
    if not install_requirements():
        return False
    
    # Setup directories
    setup_directories()
    
    print("\n🎉 Backend setup completed!")
    print("\nTo start the server:")
    print("  python run.py")
    print("\nAPI will be available at: http://localhost:8000")
    print("API docs will be available at: http://localhost:8000/docs")
    
    return True

if __name__ == "__main__":
    main()