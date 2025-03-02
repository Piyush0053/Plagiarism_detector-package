import os
import sys
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Download required NLTK data if not already downloaded
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')

def preprocess_text(text):
    """Preprocess text by tokenizing, removing stopwords, and converting to lowercase."""
    tokens = word_tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word.isalnum() and word not in stop_words]
    return ' '.join(filtered_tokens)

def read_text_file(file_path):
    """Read text from a file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return ""

def compare_files(file1_path, file2_path):
    """Compare two files for plagiarism using TF-IDF and cosine similarity."""
    text1 = read_text_file(file1_path)
    text2 = read_text_file(file2_path)
    
    if not text1 or not text2:
        return 0.0
    
    # Preprocess texts
    processed_text1 = preprocess_text(text1)
    processed_text2 = preprocess_text(text2)
    
    # Create TF-IDF vectors
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([processed_text1, processed_text2])
    
    # Calculate cosine similarity
    similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    return similarity * 100  # Convert to percentage

def check_folder_plagiarism(folder_path, file_extension='.txt'):
    """Check plagiarism among all files in a folder with the specified extension."""
    files = [f for f in os.listdir(folder_path) if f.endswith(file_extension)]
    results = {}
    
    if len(files) < 2:
        print(f"Not enough files with extension {file_extension} found in the folder.")
        return results
    
    for i in range(len(files)):
        file1 = files[i]
        file1_path = os.path.join(folder_path, file1)
        
        max_similarity = 0.0
        most_similar_file = ""
        
        for j in range(len(files)):
            if i != j:
                file2 = files[j]
                file2_path = os.path.join(folder_path, file2)
                
                similarity = compare_files(file1_path, file2_path)
                
                if similarity > max_similarity:
                    max_similarity = similarity
                    most_similar_file = file2
        
        results[file1] = (most_similar_file, f"{max_similarity:.2f}%")
    
    return results

def main():
    print("====== Simple Plagiarism Detector ======")
    print("1. Detect plagiarism in a folder of documents")
    print("2. Compare two specific files")
    print("3. Exit")
    
    choice = input("\nEnter your choice (1-3): ")
    
    if choice == '1':
        folder_path = input("Enter folder path containing documents: ")
        file_extension = input("Enter file extension (e.g., .txt, .py): ")
        
        if not file_extension.startswith('.'):
            file_extension = '.' + file_extension
        
        print("\nAnalyzing files for plagiarism...")
        results = check_folder_plagiarism(folder_path, file_extension)
        
        print("\nPlagiarism detection results:")
        for file, (similar_file, similarity) in results.items():
            print(f"File: {file}")
            print(f"Most similar to: {similar_file}")
            print(f"Similarity score: {similarity}")
            print("-------------------")
    
    elif choice == '2':
        file1 = input("Enter path to first file: ")
        file2 = input("Enter path to second file: ")
        
        print("\nComparing files for plagiarism...")
        similarity = compare_files(file1, file2)
        
        print(f"\nSimilarity between the files: {similarity:.2f}%")
        if similarity > 70:
            print("High similarity detected! Possible plagiarism.")
        elif similarity > 40:
            print("Moderate similarity detected. Some content may be shared.")
        else:
            print("Low similarity. Files appear to be different.")
    
    elif choice == '3':
        print("Exiting program.")
        sys.exit(0)
    
    else:
        print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main() 