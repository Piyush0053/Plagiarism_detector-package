import os
import pandas as pd
import numpy as np
import argparse
import random
import nltk
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
import re

# Make sure required NLTK data is downloaded
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

def load_text_files(directory):
    """Load text from all .txt files in the given directory"""
    texts = []
    
    for filename in os.listdir(directory):
        if filename.endswith('.txt'):
            filepath = os.path.join(directory, filename)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    text = f.read()
                    texts.append(text)
            except Exception as e:
                print(f"Error reading {filepath}: {e}")
    
    return texts

def create_plagiarism_pair(original_text, plagiarism_level='high'):
    """
    Create a plagiarized version of the original text
    
    Args:
        original_text (str): The original text
        plagiarism_level (str): Level of plagiarism (high, medium, low)
        
    Returns:
        str: Plagiarized text
    """
    # Tokenize into sentences
    sentences = sent_tokenize(original_text)
    
    if len(sentences) < 3:
        return original_text
        
    # Determine how much to modify based on plagiarism level
    if plagiarism_level == 'high':
        # High plagiarism: Copy 80-90% directly
        copy_rate = random.uniform(0.8, 0.9)
        synonym_replace_prob = 0.1
        sentence_reorder_prob = 0.2
    elif plagiarism_level == 'medium':
        # Medium plagiarism: Copy 50-70% directly, modify the rest
        copy_rate = random.uniform(0.5, 0.7)
        synonym_replace_prob = 0.3
        sentence_reorder_prob = 0.5
    else:  # low
        # Low plagiarism: Copy 20-40% directly, heavily modify the rest
        copy_rate = random.uniform(0.2, 0.4)
        synonym_replace_prob = 0.6
        sentence_reorder_prob = 0.8
    
    # Determine which sentences to copy directly
    num_sentences = len(sentences)
    num_to_copy = int(num_sentences * copy_rate)
    
    copy_indices = set(random.sample(range(num_sentences), num_to_copy))
    
    # Create plagiarized sentences
    plagiarized_sentences = []
    
    for i, sentence in enumerate(sentences):
        if i in copy_indices:
            # Copy directly
            plagiarized_sentences.append(sentence)
        else:
            # Modify the sentence
            words = sentence.split()
            new_words = []
            
            for word in words:
                # Simple word modification (in a real scenario, use a synonym dictionary)
                if random.random() < synonym_replace_prob and len(word) > 3:
                    # Modify the word slightly (simulating synonym replacement)
                    if word.istitle():
                        modified = word[0] + word[1:].replace('a', 'e').replace('e', 'a')
                        modified = modified.capitalize()
                    else:
                        modified = word.replace('a', 'e').replace('e', 'a')
                    new_words.append(modified)
                else:
                    new_words.append(word)
            
            modified_sentence = ' '.join(new_words)
            plagiarized_sentences.append(modified_sentence)
    
    # Possibly reorder some sentences
    if random.random() < sentence_reorder_prob and len(plagiarized_sentences) > 3:
        # Select two random sentences to swap
        idx1, idx2 = random.sample(range(len(plagiarized_sentences)), 2)
        plagiarized_sentences[idx1], plagiarized_sentences[idx2] = plagiarized_sentences[idx2], plagiarized_sentences[idx1]
    
    # Join sentences
    plagiarized_text = ' '.join(plagiarized_sentences)
    
    return plagiarized_text

def create_non_plagiarized_text(original_text, other_texts):
    """
    Create a non-plagiarized text that's different from the original
    
    Args:
        original_text (str): The original text
        other_texts (list): List of other texts to sample from
        
    Returns:
        str: Non-plagiarized text
    """
    if not other_texts:
        return create_random_text(100)  # Fallback
    
    # Sample from other texts
    sample_text = random.choice([t for t in other_texts if t != original_text])
    
    # If sample_text is too short or None, generate random text
    if not sample_text or len(sample_text.split()) < 20:
        return create_random_text(100)
    
    # Extract a portion of the sample text
    sentences = sent_tokenize(sample_text)
    
    if len(sentences) <= 3:
        return sample_text
    
    # Take a random subset of sentences
    num_sentences = min(random.randint(3, 10), len(sentences))
    selected_sentences = random.sample(sentences, num_sentences)
    
    return ' '.join(selected_sentences)

def create_random_text(word_count):
    """Generate random text with the specified word count"""
    words = ['the', 'a', 'an', 'in', 'of', 'to', 'for', 'with', 'on', 'at', 'by', 'as', 'is', 'was', 'were', 'are', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'should', 'can', 'could', 'may', 'might', 'must', 'shall', 'and', 'or', 'but', 'if', 'though', 'because', 'since', 'when', 'where', 'how', 'what', 'who', 'whom', 'which', 'whether', 'while', 'although', 'however', 'therefore', 'thus', 'hence', 'accordingly', 'consequently']
    
    nouns = ['time', 'year', 'people', 'way', 'day', 'thing', 'man', 'woman', 'life', 'world', 'school', 'state', 'family', 'student', 'group', 'country', 'problem', 'hand', 'part', 'place', 'case', 'week', 'company', 'system', 'program', 'question', 'work', 'government', 'number', 'night', 'point', 'home', 'water', 'room', 'area']
    
    adjectives = ['good', 'new', 'first', 'last', 'long', 'great', 'little', 'own', 'other', 'old', 'right', 'big', 'high', 'different', 'small', 'large', 'early', 'young', 'important', 'few', 'public', 'bad', 'same', 'able']
    
    verbs = ['be', 'have', 'do', 'say', 'go', 'can', 'get', 'would', 'make', 'know', 'will', 'think', 'take', 'see', 'come', 'could', 'want', 'look', 'use', 'find', 'give', 'tell', 'work', 'may', 'should', 'call', 'try', 'ask', 'need', 'feel', 'become', 'leave', 'put', 'mean', 'keep']
    
    generated_words = []
    
    for _ in range(word_count):
        if len(generated_words) % 8 == 0:  # Start a new sentence every 8 words
            word = random.choice(nouns).capitalize()
            generated_words.append(word)
        elif len(generated_words) % 8 == 7:  # End a sentence
            word = random.choice(verbs) + '.'
            generated_words.append(word)
        elif len(generated_words) % 8 == 1:  # After noun, usually a verb
            word = random.choice(verbs)
            generated_words.append(word)
        elif len(generated_words) % 8 == 2 or len(generated_words) % 8 == 5:  # After some verbs, article or preposition
            word = random.choice(words[:20])  # Limit to articles and prepositions
            generated_words.append(word)
        elif len(generated_words) % 8 == 3 or len(generated_words) % 8 == 6:  # After article, often adjective
            word = random.choice(adjectives)
            generated_words.append(word)
        elif len(generated_words) % 8 == 4:  # After adjective, often noun
            word = random.choice(nouns)
            generated_words.append(word)
        else:
            word = random.choice(words)
            generated_words.append(word)
    
    # Join words and ensure proper spacing after periods
    text = ' '.join(generated_words)
    text = re.sub(r'\.\s+([a-z])', lambda m: '. ' + m.group(1).upper(), text)
    
    return text

def generate_dataset(texts, output_file, num_pairs=1000):
    """
    Generate a dataset for plagiarism detection
    
    Args:
        texts (list): List of texts to use
        output_file (str): Path to output CSV file
        num_pairs (int): Number of text pairs to generate
        
    Returns:
        pd.DataFrame: Generated dataset
    """
    if not texts:
        print("No texts provided. Cannot generate dataset.")
        return None
    
    data = []
    
    # Balanced dataset with plagiarized and non-plagiarized pairs
    for i in range(num_pairs):
        # Select an original text
        original_text = random.choice(texts)
        
        # Determine if this will be a plagiarized pair
        is_plagiarized = i % 2 == 0  # 50% plagiarized, 50% non-plagiarized
        
        if is_plagiarized:
            # Create a plagiarized version of the text
            plagiarism_level = random.choice(['high', 'medium', 'low'])
            second_text = create_plagiarism_pair(original_text, plagiarism_level)
            label = 1
        else:
            # Create a non-plagiarized text
            second_text = create_non_plagiarized_text(original_text, texts)
            label = 0
        
        # Add to dataset
        data.append({
            'text1': original_text,
            'text2': second_text,
            'label': label
        })
        
        # Print progress
        if (i + 1) % 100 == 0:
            print(f"Generated {i + 1} pairs")
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Save to CSV
    df.to_csv(output_file, index=False)
    print(f"Dataset saved to {output_file}")
    
    return df

def main():
    parser = argparse.ArgumentParser(description='Generate a dataset for plagiarism detection')
    parser.add_argument('--input_dir', type=str, help='Directory containing input text files (optional)')
    parser.add_argument('--output_file', type=str, default='plagiarism_dataset.csv', help='Output CSV file')
    parser.add_argument('--num_pairs', type=int, default=1000, help='Number of text pairs to generate')
    parser.add_argument('--sample_texts', action='store_true', help='Generate sample texts if no input directory provided')
    args = parser.parse_args()
    
    texts = []
    
    # Load texts from directory if provided
    if args.input_dir and os.path.exists(args.input_dir):
        print(f"Loading texts from {args.input_dir}...")
        texts = load_text_files(args.input_dir)
        print(f"Loaded {len(texts)} texts")
    
    # Generate sample texts if no texts loaded or sample_texts flag is set
    if not texts or args.sample_texts:
        print("Generating sample texts...")
        for _ in range(50):
            text = create_random_text(random.randint(100, 500))
            texts.append(text)
    
    # Generate dataset
    print(f"Generating dataset with {args.num_pairs} pairs...")
    generate_dataset(texts, args.output_file, args.num_pairs)

if __name__ == "__main__":
    main() 