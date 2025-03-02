import os
import tempfile
import random
import re
import time
import requests
from bs4 import BeautifulSoup
from googlesearch import search
from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import uuid
# Import new packages for advanced NLP models
import torch
try:
    import gensim.downloader as api
    WORD2VEC_AVAILABLE = True
except ImportError:
    WORD2VEC_AVAILABLE = False
    print("Gensim not available. Word2Vec features will be disabled.")

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMER_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMER_AVAILABLE = False
    print("Sentence-Transformers not available. Transformer features will be disabled.")

try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("HuggingFace Transformers not available. BERT features will be disabled.")

# Import custom NLP utilities
from nlp_utils import custom_sent_tokenize, preprocess_text_for_nlp

# Import our custom model inference module
from custom_model_inference import compare_with_custom_model

# Download required NLTK data if not already downloaded
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    print("Downloading NLTK punkt tokenizer...")
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    print("Downloading NLTK stopwords...")
    nltk.download('stopwords')

# Define a custom sentence tokenizer as fallback
def custom_sent_tokenize(text):
    """Custom sentence tokenizer that uses regex as fallback if NLTK fails."""
    try:
        return sent_tokenize(text)
    except Exception as e:
        print(f"Using fallback sentence tokenizer due to: {e}")
        # Simple regex-based sentence splitting
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]

app = Flask(__name__)
app.secret_key = 'fromi_plagiarism_detector_secret_key'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB max upload size
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'  # Fix for cookie issues

# Create upload folder if it doesn't exist
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# Web-based plagiarism detection functions
def extract_random_phrases(text, num_phrases=5, phrase_length=5):
    """Extract random phrases from the document for Google search."""
    # Clean the text first to remove any non-printable characters
    clean_text = ''.join(c for c in text if c.isprintable() or c in ['\n', '\t', ' '])
    
    # Use our custom tokenizer that has fallback
    sentences = custom_sent_tokenize(clean_text)
    
    if not sentences:
        return []
    
    # Filter out very short sentences and sentences with unusual character patterns
    sentences = [s for s in sentences if len(s.split()) > phrase_length and 
                 not any(c.isalpha() for c in s) == False and  # Ensure there are alphabetic chars
                 len([c for c in s if c.isalpha()]) / len(s) > 0.5]  # At least 50% alphabetic
    
    if not sentences:
        # If no sentences meet the criteria, use the original text
        sentences = re.split(r'[.!?]+', clean_text)
        sentences = [s.strip() for s in sentences if s.strip() and len(s.split()) > 3]
    
    # Select random sentences
    if len(sentences) <= num_phrases:
        selected_sentences = sentences
    else:
        selected_sentences = random.sample(sentences, num_phrases)
    
    # Extract phrases from selected sentences
    phrases = []
    for sentence in selected_sentences:
        words = sentence.split()
        if len(words) <= phrase_length:
            phrases.append(sentence)
        else:
            start_idx = random.randint(0, len(words) - phrase_length)
            phrase = ' '.join(words[start_idx:start_idx + phrase_length])
            phrases.append(phrase)
    
    return phrases

def search_google(phrase, num_results=5):
    """Search Google for a phrase and return the top URLs."""
    try:
        print(f"Searching Google for: {phrase[:50]}...")
        search_results = list(search(phrase, num_results=num_results))
        print(f"Found {len(search_results)} results")
        return search_results
    except Exception as e:
        print(f"Error searching Google: {e}")
        # Try with a shorter phrase if the original fails
        try:
            if len(phrase.split()) > 5:
                shorter_phrase = ' '.join(phrase.split()[:5])
                print(f"Retrying with shorter phrase: {shorter_phrase}")
                search_results = list(search(shorter_phrase, num_results=num_results))
                print(f"Found {len(search_results)} results with shorter phrase")
                return search_results
        except Exception as e2:
            print(f"Error on retry: {e2}")
        return []

def scrape_content(url):
    """Scrape content from a URL."""
    try:
        # Validate URL format
        if not url.startswith(('http://', 'https://')):
            print(f"Invalid URL format: {url}")
            return ""
            
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        print(f"Sending request to: {url}")
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        print(f"Response status: {response.status_code}")
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style", "header", "footer", "nav"]):
            script.extract()
        
        # Get text
        text = soup.get_text()
        
        # Clean text
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = '\n'.join(chunk for chunk in chunks if chunk)
        
        print(f"Scraped {len(text)} characters of content")
        return text
    except requests.exceptions.RequestException as e:
        print(f"Request error scraping {url}: {e}")
        return ""
    except Exception as e:
        print(f"Error scraping {url}: {e}")
        return ""

def find_similar_segments(doc_text, web_text, threshold=0.7):
    """Find similar segments between document and web content."""
    # Use our custom tokenizer that has fallback
    doc_sentences = custom_sent_tokenize(doc_text)
    web_sentences = custom_sent_tokenize(web_text)
    
    similar_segments = []
    
    # Create TF-IDF vectors for web sentences
    if not web_sentences:
        return similar_segments
        
    try:
        web_vectorizer = TfidfVectorizer(min_df=0.0)
        web_tfidf = web_vectorizer.fit_transform(web_sentences)
        
        # For each document sentence, find similar web sentences
        for doc_sentence in doc_sentences:
            if len(doc_sentence.split()) < 5:  # Skip very short sentences
                continue
                
            doc_vector = web_vectorizer.transform([doc_sentence])
            
            # Calculate similarities
            similarities = cosine_similarity(doc_vector, web_tfidf).flatten()
            
            # Find the most similar web sentence
            max_sim_idx = np.argmax(similarities)
            max_sim = similarities[max_sim_idx]
            
            if max_sim > threshold:
                similar_segments.append({
                    'doc_text': doc_sentence,
                    'web_text': web_sentences[max_sim_idx],
                    'similarity': max_sim
                })
    except Exception as e:
        print(f"Error finding similar segments: {e}")
    
    return similar_segments

def check_web_plagiarism(text):
    """Check plagiarism by comparing document with web content."""
    # Extract random phrases
    phrases = extract_random_phrases(text)
    
    if not phrases:
        print("No phrases extracted for web search")
        return []
        
    print(f"Extracted {len(phrases)} phrases for web search")
    
    results = []
    
    for phrase in phrases:
        print(f"Searching for phrase: {phrase[:30]}...")
        # Search Google
        urls = search_google(phrase)
        
        if not urls:
            print(f"No URLs found for phrase: {phrase[:30]}...")
            continue
            
        # Filter out invalid URLs
        valid_urls = [url for url in urls if url.startswith(('http://', 'https://'))]
        
        if not valid_urls:
            print(f"No valid URLs found for phrase: {phrase[:30]}...")
            continue
            
        print(f"Found {len(valid_urls)} valid URLs for phrase")
        
        for url in valid_urls:
            print(f"Scraping content from: {url}")
            # Scrape content
            web_content = scrape_content(url)
            
            if not web_content:
                print(f"No content scraped from: {url}")
                continue
            
            print(f"Scraped {len(web_content)} characters from {url}")
            
            # Find similar segments
            similar_segments = find_similar_segments(text, web_content)
            
            if similar_segments:
                print(f"Found {len(similar_segments)} similar segments")
                # Calculate overall similarity
                similarity_scores = [seg['similarity'] for seg in similar_segments]
                overall_similarity = sum(similarity_scores) / len(similarity_scores) if similarity_scores else 0
                
                results.append({
                    'url': url,
                    'similarity': overall_similarity * 100,  # Convert to percentage
                    'similar_segments': similar_segments
                })
            else:
                print(f"No similar segments found for {url}")
            
            # Be nice to Google and avoid rate limiting
            time.sleep(2)
    
    # Sort results by similarity (highest first)
    results.sort(key=lambda x: x['similarity'], reverse=True)
    
    return results

def preprocess_text(text):
    """Preprocess text by tokenizing, removing stopwords, and converting to lowercase."""
    # Simple tokenization by splitting on whitespace instead of using word_tokenize
    tokens = text.lower().split()
    stop_words = set(stopwords.words('english'))
    # Keep some common words to avoid empty vocabulary
    filtered_tokens = [word for word in tokens if word.isalnum() or word.isalpha()]
    
    # If after filtering we have no tokens left, return the original lowercase text
    if not filtered_tokens:
        return text.lower()
        
    return ' '.join(filtered_tokens)

def read_text_file(file_path):
    """Read text from a file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except UnicodeDecodeError:
        # Try with a different encoding if UTF-8 fails
        try:
            with open(file_path, 'r', encoding='latin-1') as file:
                return file.read()
        except UnicodeDecodeError:
            # If still can't decode, might be a binary file
            return extract_text_from_binary(file_path)
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return ""

def extract_text_from_binary(file_path):
    """Try to extract text from binary files."""
    try:
        # Read as binary and try to find text content
        with open(file_path, 'rb') as file:
            content = file.read()
            
        # Try to decode with errors='ignore' to skip non-text parts
        text = content.decode('utf-8', errors='ignore')
        
        # Clean up the text by removing non-printable characters
        printable_text = ''.join(c for c in text if c.isprintable() or c in ['\n', '\t', ' '])
        
        return printable_text
    except Exception as e:
        print(f"Error extracting text from binary file {file_path}: {e}")
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
        return results, "Not enough files with the specified extension found."
    
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
    
    return results, None

def compare_text_content(text1, text2):
    """Compare two text contents for plagiarism using TF-IDF and cosine similarity."""
    if not text1 or not text2:
        return 0.0
    
    # Preprocess texts
    processed_text1 = preprocess_text(text1)
    processed_text2 = preprocess_text(text2)
    
    try:
        # Create TF-IDF vectors
        vectorizer = TfidfVectorizer(min_df=0.0)
        tfidf_matrix = vectorizer.fit_transform([processed_text1, processed_text2])
        
        # Calculate cosine similarity
        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        return similarity * 100  # Convert to percentage
    except ValueError as e:
        # In case of empty vocabulary or other vectorizer errors
        print(f"Error in vectorization: {e}")
        # Fallback to a simple character-level comparison if TF-IDF fails
        common_chars = set(processed_text1) & set(processed_text2)
        all_chars = set(processed_text1) | set(processed_text2)
        
        if not all_chars:
            return 0.0
            
        return len(common_chars) / len(all_chars) * 100

def save_uploaded_files(files):
    """Save uploaded files to the upload folder and return their paths."""
    file_paths = []
    for file in files:
        if file.filename:
            filename = str(uuid.uuid4()) + '_' + file.filename
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            file_paths.append(file_path)
    return file_paths

def save_text_content_to_file(text_content):
    """Save text content to a temporary file and return the file path."""
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], str(uuid.uuid4()) + '.txt')
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(text_content)
    return file_path

@app.route('/')
def index():
    """Render the home page."""
    return render_template('index.html')

@app.route('/compare-files', methods=['POST'])
def compare_files_route():
    """Compare two uploaded files for plagiarism."""
    if 'file1' not in request.files or 'file2' not in request.files:
        flash('Please upload both files.', 'warning')
        return redirect(url_for('index'))
    
    file1 = request.files['file1']
    file2 = request.files['file2']
    
    if not file1.filename or not file2.filename:
        flash('Please select both files.', 'warning')
        return redirect(url_for('index'))
    
    file1_path = os.path.join(app.config['UPLOAD_FOLDER'], str(uuid.uuid4()) + '_' + file1.filename)
    file2_path = os.path.join(app.config['UPLOAD_FOLDER'], str(uuid.uuid4()) + '_' + file2.filename)
    
    file1.save(file1_path)
    file2.save(file2_path)
    
    similarity = compare_files(file1_path, file2_path)
    
    # Clean up uploaded files
    try:
        os.remove(file1_path)
        os.remove(file2_path)
    except:
        pass
    
    result = {
        'file1': file1.filename,
        'file2': file2.filename,
        'similarity': f"{similarity:.2f}%",
        'plagiarism_level': "High" if similarity > 70 else "Moderate" if similarity > 40 else "Low"
    }
    
    return render_template('results.html', result=result, mode='compare_files')

@app.route('/compare-text', methods=['POST'])
def compare_text_route():
    """Compare two text inputs for plagiarism."""
    text1 = request.form.get('text1', '')
    text2 = request.form.get('text2', '')
    
    if not text1 or not text2:
        flash('Please enter both text contents.', 'warning')
        return redirect(url_for('index'))
    
    similarity = compare_text_content(text1, text2)
    
    result = {
        'similarity': f"{similarity:.2f}%",
        'plagiarism_level': "High" if similarity > 70 else "Moderate" if similarity > 40 else "Low"
    }
    
    return render_template('results.html', result=result, mode='compare_text', text1=text1, text2=text2)

@app.route('/check-folder', methods=['POST'])
def check_folder_route():
    """Check plagiarism in uploaded files."""
    if 'folder' not in request.files:
        flash('Please upload at least two files.', 'warning')
        return redirect(url_for('index'))
    
    uploaded_files = request.files.getlist('folder')
    
    if len(uploaded_files) < 2 or not uploaded_files[0].filename or not uploaded_files[1].filename:
        flash('Please upload at least two files.', 'warning')
        return redirect(url_for('index'))
    
    # Create a temporary directory to store uploaded files
    temp_dir = os.path.join(app.config['UPLOAD_FOLDER'], str(uuid.uuid4()))
    os.makedirs(temp_dir, exist_ok=True)
    
    file_paths = []
    for file in uploaded_files:
        if file.filename:
            file_path = os.path.join(temp_dir, file.filename)
            file.save(file_path)
            file_paths.append(file_path)
    
    # Get file extension from the first file
    _, file_extension = os.path.splitext(file_paths[0])
    
    results, error = check_folder_plagiarism(temp_dir, file_extension)
    
    # Clean up uploaded files
    for file_path in file_paths:
        try:
            os.remove(file_path)
        except:
            pass
    try:
        os.rmdir(temp_dir)
    except:
        pass
    
    if error:
        flash(error, 'warning')
        return redirect(url_for('index'))
    
    return render_template('results.html', results=results, mode='check_folder')

@app.route('/check-web-plagiarism', methods=['POST'])
def check_web_plagiarism_route():
    """Check plagiarism by comparing uploaded document with web content."""
    if 'file' not in request.files:
        flash('Please upload a file.', 'warning')
        return redirect(url_for('index'))
    
    file = request.files['file']
    
    if not file.filename:
        flash('Please select a file.', 'warning')
        return redirect(url_for('index'))
    
    try:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], str(uuid.uuid4()) + '_' + file.filename)
        file.save(file_path)
        
        # Read file content
        text = read_text_file(file_path)
        
        if not text or len(text.strip()) < 50:
            flash('The uploaded file is empty or too short for meaningful plagiarism detection.', 'warning')
            return redirect(url_for('index'))
        
        flash('Processing your document for web plagiarism check. This may take a minute...', 'info')
        
        # Check web plagiarism
        results = check_web_plagiarism(text)
        
        # Clean up uploaded file
        try:
            os.remove(file_path)
        except Exception as e:
            print(f"Error removing file: {e}")
        
        if not results:
            flash('No plagiarism detected or unable to perform web search. Try with different content.', 'info')
        
        return render_template('results.html', web_results=results, mode='web_plagiarism', filename=file.filename)
    except Exception as e:
        print(f"Error in web plagiarism check: {e}")
        flash(f'An error occurred during plagiarism detection: {str(e)}', 'danger')
        return redirect(url_for('index'))

@app.route('/check-text-web-plagiarism', methods=['POST'])
def check_text_web_plagiarism_route():
    """Check plagiarism by comparing text input with web content."""
    text = request.form.get('text', '')
    
    if not text:
        flash('Please enter text content.', 'warning')
        return redirect(url_for('index'))
    
    if len(text.strip()) < 50:
        flash('The text is too short for meaningful plagiarism detection. Please enter at least 50 characters.', 'warning')
        return redirect(url_for('index'))
    
    try:
        flash('Processing your text for web plagiarism check. This may take a minute...', 'info')
        
        # Check web plagiarism
        results = check_web_plagiarism(text)
        
        if not results:
            flash('No plagiarism detected or unable to perform web search. Try with different content.', 'info')
        
        return render_template('results.html', web_results=results, mode='web_plagiarism_text', text=text)
    except Exception as e:
        print(f"Error in text web plagiarism check: {e}")
        flash(f'An error occurred during plagiarism detection: {str(e)}', 'danger')
        return redirect(url_for('index'))

# Advanced NLP model functions
def load_word2vec_model():
    """Load Word2Vec model if available."""
    if WORD2VEC_AVAILABLE:
        try:
            # Load a smaller model for efficiency
            return api.load('glove-wiki-gigaword-100')
        except Exception as e:
            print(f"Error loading Word2Vec model: {e}")
            return None
    return None

def load_sentence_transformer_model():
    """Load Sentence Transformer model if available."""
    if SENTENCE_TRANSFORMER_AVAILABLE:
        try:
            # Load a smaller, efficient model
            return SentenceTransformer('all-MiniLM-L6-v2')
        except Exception as e:
            print(f"Error loading Sentence Transformer model: {e}")
            return None
    return None

def load_paraphrase_model():
    """Load paraphrase detection model if available."""
    if TRANSFORMERS_AVAILABLE:
        try:
            tokenizer = AutoTokenizer.from_pretrained("microsoft/mpnet-base-v2")
            model = AutoModelForSequenceClassification.from_pretrained("cross-encoder/stsb-roberta-large")
            return tokenizer, model
        except Exception as e:
            print(f"Error loading paraphrase model: {e}")
            return None, None
    return None, None

# Initialize models (loading them lazily when needed)
word2vec_model = None
sentence_transformer_model = None
paraphrase_tokenizer, paraphrase_model = None, None

def compare_with_word_embeddings(text1, text2):
    """Compare two texts using Word2Vec embeddings."""
    global word2vec_model
    
    if not WORD2VEC_AVAILABLE:
        return 0.0
    
    if word2vec_model is None:
        word2vec_model = load_word2vec_model()
        if word2vec_model is None:
            return 0.0
    
    # Tokenize texts
    tokens1 = preprocess_text(text1).split()
    tokens2 = preprocess_text(text2).split()
    
    # Get embeddings for each word
    embeddings1 = [word2vec_model[word] for word in tokens1 if word in word2vec_model]
    embeddings2 = [word2vec_model[word] for word in tokens2 if word in word2vec_model]
    
    if not embeddings1 or not embeddings2:
        return 0.0
    
    # Calculate document vectors (average of word vectors)
    doc_vector1 = np.mean(embeddings1, axis=0)
    doc_vector2 = np.mean(embeddings2, axis=0)
    
    # Calculate cosine similarity
    similarity = cosine_similarity([doc_vector1], [doc_vector2])[0][0]
    return similarity * 100

def compare_with_sentence_transformer(text1, text2):
    """Compare two texts using Sentence Transformers."""
    global sentence_transformer_model
    
    if not SENTENCE_TRANSFORMER_AVAILABLE:
        return 0.0
    
    if sentence_transformer_model is None:
        sentence_transformer_model = load_sentence_transformer_model()
        if sentence_transformer_model is None:
            return 0.0
    
    # Split texts into sentences
    sentences1 = custom_sent_tokenize(text1)
    sentences2 = custom_sent_tokenize(text2)
    
    if not sentences1 or not sentences2:
        return 0.0
    
    try:
        # Get embeddings
        embeddings1 = sentence_transformer_model.encode(sentences1)
        embeddings2 = sentence_transformer_model.encode(sentences2)
        
        # Find best matches between sentences
        similarities = []
        for emb1 in embeddings1:
            # Calculate similarity with each sentence in text2
            scores = cosine_similarity([emb1], embeddings2)[0]
            # Get highest similarity score
            max_sim = np.max(scores)
            similarities.append(max_sim)
        
        # Average the similarities
        avg_similarity = np.mean(similarities) if similarities else 0.0
        return avg_similarity * 100
    except Exception as e:
        print(f"Error in sentence transformer comparison: {e}")
        return 0.0

def detect_paraphrases(text1, text2, threshold=0.7):
    """Detect paraphrased content between two texts using BERT."""
    global paraphrase_tokenizer, paraphrase_model
    
    if not TRANSFORMERS_AVAILABLE:
        return []
    
    if paraphrase_tokenizer is None or paraphrase_model is None:
        paraphrase_tokenizer, paraphrase_model = load_paraphrase_model()
        if paraphrase_tokenizer is None or paraphrase_model is None:
            return []
    
    # Split texts into sentences
    sentences1 = custom_sent_tokenize(text1)
    sentences2 = custom_sent_tokenize(text2)
    
    results = []
    
    try:
        # Compare each sentence from text1 with each from text2
        for sent1 in sentences1:
            if len(sent1.split()) < 5:  # Skip very short sentences
                continue
                
            for sent2 in sentences2:
                if len(sent2.split()) < 5:  # Skip very short sentences
                    continue
                    
                # Tokenize
                inputs = paraphrase_tokenizer(sent1, sent2, padding=True, truncation=True, return_tensors="pt")
                
                # Get prediction
                with torch.no_grad():
                    outputs = paraphrase_model(**inputs)
                    scores = outputs.logits.flatten()
                    
                similarity = torch.sigmoid(scores).item()  # Convert to probability
                
                if similarity > threshold:  # Threshold for paraphrase detection
                    results.append({
                        'text1': sent1,
                        'text2': sent2,
                        'similarity': similarity
                    })
        
        return results
    except Exception as e:
        print(f"Error in paraphrase detection: {e}")
        return []

def multilingual_similarity(text1, text2, language1="en", language2="en"):
    """Compare texts across different languages using multilingual model."""
    global sentence_transformer_model
    
    if not SENTENCE_TRANSFORMER_AVAILABLE:
        return 0.0
    
    # For multilingual, we need a specific model
    try:
        # Use a multilingual model
        if sentence_transformer_model is None or "multilingual" not in str(sentence_transformer_model):
            # Only load the multilingual model when needed
            multilingual_model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
        else:
            multilingual_model = sentence_transformer_model
        
        # Encode texts
        embedding1 = multilingual_model.encode(text1)
        embedding2 = multilingual_model.encode(text2)
        
        # Calculate similarity
        similarity = cosine_similarity([embedding1], [embedding2])[0][0]
        return similarity * 100
    except Exception as e:
        print(f"Error in multilingual comparison: {e}")
        return 0.0

def advanced_compare_text_content(text1, text2):
    """Compare two texts using all available NLP models and return combined results."""
    results = {}
    
    # Standard TF-IDF similarity
    tfidf_similarity = compare_text_content(text1, text2)
    results['tfidf_similarity'] = tfidf_similarity
    
    # Word embeddings similarity
    if WORD2VEC_AVAILABLE:
        word_embedding_similarity = compare_with_word_embeddings(text1, text2)
        results['word_embedding_similarity'] = word_embedding_similarity
    
    # Sentence transformer similarity
    if SENTENCE_TRANSFORMER_AVAILABLE:
        transformer_similarity = compare_with_sentence_transformer(text1, text2)
        results['transformer_similarity'] = transformer_similarity
    
    # Paraphrase detection
    if TRANSFORMERS_AVAILABLE:
        paraphrases = detect_paraphrases(text1, text2)
        results['paraphrases'] = paraphrases
        if paraphrases:
            # Calculate average similarity of detected paraphrases
            avg_paraphrase_sim = sum([p['similarity'] for p in paraphrases]) / len(paraphrases) * 100
            results['paraphrase_similarity'] = avg_paraphrase_sim
    
    # Calculate combined score
    # Weight more advanced models higher
    weights = {
        'tfidf_similarity': 1.0,
        'word_embedding_similarity': 1.5,
        'transformer_similarity': 2.0,
        'paraphrase_similarity': 2.5
    }
    
    total_weight = 0
    weighted_sum = 0
    
    for metric, weight in weights.items():
        if metric in results:
            weighted_sum += results[metric] * weight
            total_weight += weight
    
    if total_weight > 0:
        results['combined_similarity'] = weighted_sum / total_weight
    else:
        results['combined_similarity'] = tfidf_similarity  # Fall back to TF-IDF
    
    return results

# New routes for advanced plagiarism detection
@app.route('/advanced-compare-text', methods=['POST'])
def advanced_compare_text_route():
    """Compare two text inputs using advanced NLP models."""
    text1 = request.form.get('text1', '')
    text2 = request.form.get('text2', '')
    
    if not text1 or not text2:
        flash('Please enter both text contents.', 'warning')
        return redirect(url_for('index'))
    
    # Get advanced similarity results
    results = advanced_compare_text_content(text1, text2)
    
    # Format results for display
    display_results = {
        'text1': text1,
        'text2': text2,
        'tfidf_similarity': f"{results.get('tfidf_similarity', 0.0):.2f}%",
        'combined_similarity': f"{results.get('combined_similarity', 0.0):.2f}%",
        'plagiarism_level': "High" if results.get('combined_similarity', 0.0) > 70 else 
                          "Moderate" if results.get('combined_similarity', 0.0) > 40 else "Low"
    }
    
    # Add optional results if available
    if 'word_embedding_similarity' in results:
        display_results['word_embedding_similarity'] = f"{results['word_embedding_similarity']:.2f}%"
    
    if 'transformer_similarity' in results:
        display_results['transformer_similarity'] = f"{results['transformer_similarity']:.2f}%"
    
    if 'paraphrase_similarity' in results:
        display_results['paraphrase_similarity'] = f"{results['paraphrase_similarity']:.2f}%"
    
    if 'paraphrases' in results and results['paraphrases']:
        display_results['paraphrases'] = results['paraphrases']
    
    return render_template('advanced_results.html', result=display_results, mode='advanced_compare_text')

@app.route('/advanced-compare-files', methods=['POST'])
def advanced_compare_files_route():
    """Compare two uploaded files using advanced NLP models."""
    if 'file1' not in request.files or 'file2' not in request.files:
        flash('Please upload both files.', 'warning')
        return redirect(url_for('index'))
    
    file1 = request.files['file1']
    file2 = request.files['file2']
    
    if not file1.filename or not file2.filename:
        flash('Please select both files.', 'warning')
        return redirect(url_for('index'))
    
    file1_path = os.path.join(app.config['UPLOAD_FOLDER'], str(uuid.uuid4()) + '_' + file1.filename)
    file2_path = os.path.join(app.config['UPLOAD_FOLDER'], str(uuid.uuid4()) + '_' + file2.filename)
    
    file1.save(file1_path)
    file2.save(file2_path)
    
    # Read file contents
    text1 = read_text_file(file1_path)
    text2 = read_text_file(file2_path)
    
    # Get advanced similarity results
    results = advanced_compare_text_content(text1, text2)
    
    # Clean up uploaded files
    try:
        os.remove(file1_path)
        os.remove(file2_path)
    except:
        pass
    
    # Format results for display
    display_results = {
        'file1': file1.filename,
        'file2': file2.filename,
        'tfidf_similarity': f"{results.get('tfidf_similarity', 0.0):.2f}%",
        'combined_similarity': f"{results.get('combined_similarity', 0.0):.2f}%",
        'plagiarism_level': "High" if results.get('combined_similarity', 0.0) > 70 else 
                          "Moderate" if results.get('combined_similarity', 0.0) > 40 else "Low"
    }
    
    # Add optional results if available
    if 'word_embedding_similarity' in results:
        display_results['word_embedding_similarity'] = f"{results['word_embedding_similarity']:.2f}%"
    
    if 'transformer_similarity' in results:
        display_results['transformer_similarity'] = f"{results['transformer_similarity']:.2f}%"
    
    if 'paraphrase_similarity' in results:
        display_results['paraphrase_similarity'] = f"{results['paraphrase_similarity']:.2f}%"
    
    if 'paraphrases' in results and results['paraphrases']:
        display_results['paraphrases'] = results['paraphrases']
    
    return render_template('advanced_results.html', result=display_results, mode='advanced_compare_files')

@app.route('/multilingual-compare-text', methods=['POST'])
def multilingual_compare_text_route():
    """Compare texts in different languages."""
    text1 = request.form.get('text1', '')
    text2 = request.form.get('text2', '')
    language1 = request.form.get('language1', 'en')
    language2 = request.form.get('language2', 'en')
    
    if not text1 or not text2:
        flash('Please enter both text contents.', 'warning')
        return redirect(url_for('index'))
    
    # Get multilingual similarity
    similarity = multilingual_similarity(text1, text2, language1, language2)
    
    result = {
        'text1': text1,
        'text2': text2,
        'language1': language1,
        'language2': language2,
        'similarity': f"{similarity:.2f}%",
        'plagiarism_level': "High" if similarity > 70 else "Moderate" if similarity > 40 else "Low"
    }
    
    return render_template('advanced_results.html', result=result, mode='multilingual_compare_text')

@app.route('/custom-model-compare', methods=['POST'])
def custom_model_compare():
    """Compare texts using the custom-trained plagiarism detection model"""
    if request.method == 'POST':
        # Get texts from request
        text1 = request.form.get('text1', '')
        text2 = request.form.get('text2', '')
        
        if not text1 or not text2:
            return jsonify({
                'error': 'Both text fields are required',
                'score': 0,
                'is_plagiarized': False
            })
        
        # Set threshold (optional, default is 0.5)
        threshold = float(request.form.get('threshold', 0.5))
        
        # Compare texts using custom model
        result = compare_with_custom_model(text1, text2, threshold=threshold)
        
        # Convert score to percentage for UI
        if 'score' in result:
            result['score_percent'] = round(result['score'] * 100, 2)
        
        # Return results
        return jsonify(result)

@app.route('/train-custom-model', methods=['POST'])
def train_custom_model():
    """Endpoint to trigger custom model training"""
    # This would typically be a long-running process that should be run asynchronously
    # For demonstration, we'll just return a message
    return jsonify({
        'status': 'success',
        'message': 'Model training initiated. This process will run in the background.',
        'info': 'In a production environment, this would trigger an asynchronous training job.'
    })

@app.route('/custom-model-status', methods=['GET'])
def custom_model_status():
    """Check if the custom model is available"""
    model_dir = os.path.join(app.root_path, 'model')
    model_path = os.path.join(model_dir, 'plagiarism_detector.pt')
    
    if os.path.exists(model_path):
        return jsonify({
            'status': 'available',
            'message': 'Custom plagiarism detection model is available.'
        })
    else:
        return jsonify({
            'status': 'unavailable',
            'message': 'Custom plagiarism detection model is not available. Please train the model first.'
        })

@app.route('/custom-model')
def custom_model():
    """Display the custom model page"""
    return render_template('custom_model.html')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000) 