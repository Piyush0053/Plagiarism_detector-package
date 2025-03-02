"""
Run the FROMI Plagiarism Detector Web Application
This script launches the Flask web server for the plagiarism detection application.
"""

import os
import sys
import subprocess
import webbrowser
from time import sleep

def check_dependencies():
    """Check if all required dependencies are installed."""
    try:
        import nltk
        import sklearn
        import flask
        
        # Check if NLTK data is downloaded
        try:
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('corpora/stopwords')
        except LookupError:
            print("Downloading required NLTK data...")
            nltk.download('punkt')
            nltk.download('stopwords')
            print("NLTK data downloaded successfully.")
            
        return True
    except ImportError as e:
        print(f"Missing dependency: {str(e)}")
        print("Please install all dependencies first with: pip install -r requirements.txt")
        return False

def run_application():
    """Run the Flask web application."""
    try:
        print("Starting FROMI Plagiarism Detector Web Application...")
        print("The web interface will be available at: http://localhost:5000")
        
        # Open the web browser after a short delay
        def open_browser():
            sleep(1.5)  # Give the server a moment to start
            webbrowser.open('http://localhost:5000')
        
        # Start browser in a separate thread
        import threading
        browser_thread = threading.Thread(target=open_browser)
        browser_thread.daemon = True
        browser_thread.start()
        
        # Change to the web_app directory
        web_app_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'web_app')
        os.chdir(web_app_dir)
        
        # Run the Flask application
        from web_app.app import app
        app.run(debug=True, threaded=True)
        
    except Exception as e:
        print(f"Error starting the application: {str(e)}")
        return False
    
    return True

if __name__ == "__main__":
    print("=" * 60)
    print("FROMI Plagiarism Detector Web Application")
    print("=" * 60)
    
    if check_dependencies():
        run_application()
    else:
        sys.exit(1)