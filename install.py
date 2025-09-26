"""
Installation script for YOLO Camera Detection Studio
Installs dependencies one by one for better compatibility
"""

import subprocess
import sys

def install_package(package_name):
    """Install a single package using pip"""
    try:
        print(f"Installing {package_name}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
        print(f"✓ {package_name} installed successfully\n")
        return True
    except subprocess.CalledProcessError:
        print(f"✗ Failed to install {package_name}\n")
        return False

def main():
    print("=" * 60)
    print("YOLO Krypton - Dependency Installer")
    print("=" * 60)
    print()
    
    # List of packages to install
    packages = [
        "customtkinter",
        "Pillow",
        "numpy",
        "opencv-python",
        "ultralytics",
        "matplotlib",
        "pandas"
    ]
    
    failed_packages = []
    
    # Install each package
    for package in packages:
        if not install_package(package):
            failed_packages.append(package)
    
    print("=" * 60)
    
    if failed_packages:
        print("⚠ The following packages failed to install:")
        for package in failed_packages:
            print(f"  - {package}")
        print("\nYou may need to install them manually or check for compatibility issues.")
        print("Try running: pip install <package_name> --upgrade")
    else:
        print("✓ All packages installed successfully!")
        print("\nYou can now run the application with: python main.py")
    
    print("=" * 60)

if __name__ == "__main__":
    main()
