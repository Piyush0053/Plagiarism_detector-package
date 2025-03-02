import os
import sys
import traceback

# Add the packages directory to Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)

# Import the plagiarism detection modules
try:
    from packages.plag import plag
    from packages.imageplag import imageplag
    from packages.cmplag import cmplag
    print("Successfully imported plagiarism modules")
except ImportError as e:
    print(f"Error importing modules: {e}")
    traceback.print_exc()
    print("Make sure all dependencies are installed by running: pip install -r requirements.txt")
    sys.exit(1)

def main():
    try:
        print("====== FROMI Plagiarism Detector ======")
        print("1. Detect text plagiarism in a folder of documents")
        print("2. Check plagiarism in specific files")
        print("3. Image plagiarism detection")
        print("4. Exit")
        
        choice = input("\nEnter your choice (1-4): ")
        
        if choice == '1':
            folder_path = input("Enter folder path containing documents: ")
            file_type = input("Enter file type (pdf/doc/txt): ")
            
            print("\nAnalyzing files for plagiarism...")
            result = plag.mutualfileplag(folder_path, file_type)
            print("\nPlagiarism detection results:")
            for file, data in result.items():
                print(f"File: {file}")
                print(f"Similar to: {data[0]}")
                print(f"Similarity score: {data[1]}")
                print("-------------------")
        
        elif choice == '2':
            file1 = input("Enter path to first file: ")
            file2 = input("Enter path to second file: ")
            
            print("\nComparing files for plagiarism...")
            result = cmplag.mutualchecker(file1, file2)
            print(f"The files have been analyzed. Check the highlighted content for similarities.")
        
        elif choice == '3':
            image_file = input("Enter path to image file: ")
            
            print("\nAnalyzing image for plagiarism...")
            result = imageplag.imageinfo(image_file)
            print("Image analysis completed.")
        
        elif choice == '4':
            print("Exiting program.")
            sys.exit(0)
        
        else:
            print("Invalid choice. Please try again.")
    except Exception as e:
        print(f"An error occurred: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main() 