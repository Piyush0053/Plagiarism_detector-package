import os
import sys
import subprocess
import importlib.util

def check_module_installed(module_name):
    """Check if a Python module is installed."""
    return importlib.util.find_spec(module_name) is not None

def install_requirements():
    """Install required Python packages from requirements.txt."""
    print("Installing required Python packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("Requirements installed successfully.")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error installing requirements: {e}")
        return False

def download_nltk_data():
    """Download required NLTK data."""
    print("Downloading required NLTK data...")
    try:
        import nltk
        nltk.download('punkt')
        nltk.download('stopwords')
        nltk.download('wordnet')
        print("NLTK data downloaded successfully.")
        return True
    except Exception as e:
        print(f"Error downloading NLTK data: {e}")
        return False

def check_tesseract():
    """Check if Tesseract OCR is installed."""
    try:
        import pytesseract
        pytesseract.get_tesseract_version()
        print("Tesseract OCR is installed.")
        return True
    except Exception:
        print("Warning: Tesseract OCR may not be installed or configured properly.")
        print("Image-based plagiarism detection may not work correctly.")
        print("Please download and install Tesseract OCR:")
        print("- Windows: https://github.com/UB-Mannheim/tesseract/wiki")
        print("- Linux: sudo apt-get install tesseract-ocr")
        print("- macOS: brew install tesseract")
        return False

def main():
    print("===== Initializing FROMI Plagiarism Detector =====")
    
    # Check and install requirements
    if not install_requirements():
        print("Failed to install requirements. Please install them manually.")
        return False
    
    # Download NLTK data
    if not download_nltk_data():
        print("Failed to download NLTK data. Some features might not work correctly.")
    
    # Check Tesseract OCR
    check_tesseract()
    
    print("\nInitialization completed. You can now run the plagiarism detector with:")
    print("python run_plagiarism_detector.py")
    
    return True

if __name__ == "__main__":
    main() 