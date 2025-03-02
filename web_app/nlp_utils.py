import re
import nltk
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords

# Make sure required NLTK data is downloaded
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

def custom_sent_tokenize(text):
    """
    Custom sentence tokenizer with additional cleaning and filtering
    
    Args:
        text (str): The input text to tokenize into sentences
        
    Returns:
        list: A list of cleaned, filtered sentences
    """
    if not text or not isinstance(text, str):
        return []
    
    # Basic cleaning
    text = text.replace('\n', ' ').replace('\r', ' ')
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Try to tokenize using NLTK
    try:
        sentences = sent_tokenize(text)
    except Exception as e:
        print(f"Error in sentence tokenization: {e}")
        # Fallback: split by common sentence endings
        sentences = re.split(r'(?<=[.!?])\s+', text)
    
    # Filter and clean sentences
    cleaned_sentences = []
    for sentence in sentences:
        # Skip very short sentences
        if len(sentence.split()) < 3:
            continue
            
        # Skip sentences with unusual character patterns (might be code or non-text)
        if re.search(r'[^\w\s.,!?;:\'"-]', sentence) and len(re.findall(r'[^\w\s.,!?;:\'"-]', sentence)) > len(sentence) * 0.3:
            continue
            
        # Clean the sentence
        cleaned = sentence.strip()
        if cleaned:
            cleaned_sentences.append(cleaned)
    
    return cleaned_sentences

def preprocess_text_for_nlp(text):
    """
    Preprocess text for NLP models by cleaning, tokenizing, and removing stopwords
    
    Args:
        text (str): The input text to preprocess
        
    Returns:
        str: Preprocessed text
    """
    if not text or not isinstance(text, str):
        return ""
    
    # Lowercase
    text = text.lower()
    
    # Remove special characters and digits
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Tokenize and remove stopwords
    try:
        stop_words = set(stopwords.words('english'))
        words = text.split()
        words = [word for word in words if word not in stop_words]
        return ' '.join(words)
    except Exception as e:
        print(f"Error in text preprocessing: {e}")
        return text 