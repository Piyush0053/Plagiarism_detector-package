import os
import pandas as pd
import argparse
import random
from tqdm import tqdm
import itertools

def download_daigt_dataset(output_dir="dataset"):
    """
    Download DAIGT dataset from Kaggle. 
    Note: This requires kaggle API credentials to be set up.
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
        
        print("Attempting to download DAIGT dataset using Kaggle API...")
        os.system(f"kaggle datasets download thedrcat/daigt-v2-train-dataset -p {output_dir} --unzip")
        
        # Check if files were downloaded successfully
        csv_files = [f for f in os.listdir(output_dir) if f.endswith('.csv')]
        if not csv_files:
            print("Warning: No CSV files found after download. Check your Kaggle API credentials.")
        else:
            print(f"Successfully downloaded {len(csv_files)} CSV files.")
            for f in csv_files:
                print(f" - {f}")
        
        return True
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        print("Manual download instructions:")
        print("1. Go to https://www.kaggle.com/datasets/thedrcat/daigt-v2-train-dataset")
        print("2. Click 'Download'")
        print(f"3. Extract the files to the '{output_dir}' directory")
        return False

def convert_to_plagiarism_format(input_dir="dataset", output_file="dataset/plagiarism_dataset.csv", sample_size=None):
    """
    Convert DAIGT dataset to the format used by our plagiarism detector.
    
    The DAIGT dataset contains essays with a "generated" label indicating if they were AI-generated.
    We'll create pairs of essays, with some being similar (plagiarism) and others different.
    
    Args:
        input_dir: Directory containing DAIGT dataset files
        output_file: Path to output CSV file
        sample_size: Number of pairs to generate (if None, use all possible pairs)
    """
    csv_files = [f for f in os.listdir(input_dir) if f.endswith('.csv')]
    
    if not csv_files:
        print(f"No CSV files found in {input_dir}")
        return False
    
    # Load and combine data from all CSV files
    all_data = []
    for file in csv_files:
        try:
            df = pd.read_csv(os.path.join(input_dir, file))
            # Add source file name as a column
            df['source_file'] = file
            all_data.append(df)
            print(f"Loaded {len(df)} rows from {file}")
        except Exception as e:
            print(f"Error loading {file}: {e}")
    
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Print dataset statistics
    print(f"Loaded {len(combined_df)} essays from {len(csv_files)} files")
    
    # Check if the dataset has the expected columns
    if 'text' not in combined_df.columns:
        print("Error: Dataset does not contain 'text' column")
        if 'full_text' in combined_df.columns:
            print("Found 'full_text' column, using that instead")
            combined_df['text'] = combined_df['full_text']
        else:
            print(f"Available columns: {combined_df.columns.tolist()}")
            return False
    
    # Use 'label' column if available, otherwise try 'generated', or use random labels
    if 'generated' in combined_df.columns:
        print("Using 'generated' column for labels")
    elif 'label' in combined_df.columns:
        print("Using 'label' column for AI-generated detection")
        combined_df['generated'] = combined_df['label']
    else:
        print("Warning: Dataset does not contain 'generated' or 'label' column, using random labels")
        combined_df['generated'] = [random.choice([0, 1]) for _ in range(len(combined_df))]
    
    # Limit dataset size for memory efficiency
    max_essays = 2000 if sample_size else 5000
    if len(combined_df) > max_essays:
        print(f"Limiting dataset to {max_essays} essays for memory efficiency")
        combined_df = combined_df.sample(max_essays, random_state=42)
    
    # Create plagiarism pairs
    data = []
    
    # Positive pairs (plagiarism) - essays with same generation status
    # These are considered similar (plagiarized from same source)
    generated = combined_df[combined_df['generated'] == 1]
    human = combined_df[combined_df['generated'] == 0]
    
    print(f"Found {len(generated)} AI-generated essays and {len(human)} human-written essays")
    
    # Limit the size of each group for memory efficiency
    max_group_size = 500
    if len(generated) > max_group_size:
        generated = generated.sample(max_group_size, random_state=42)
    if len(human) > max_group_size:
        human = human.sample(max_group_size, random_state=42)
    
    # Create positive pairs from generated essays
    print("Creating positive pairs from generated essays...")
    generated_pairs = create_pairs_from_group(generated, sample_size//4 if sample_size else 1000, label=1)
    data.extend(generated_pairs)
    
    # Create positive pairs from human essays
    print("Creating positive pairs from human essays...")
    human_pairs = create_pairs_from_group(human, sample_size//4 if sample_size else 1000, label=1)
    data.extend(human_pairs)
    
    # Create negative pairs (non-plagiarism) - one generated, one human
    print("Creating negative pairs (human vs generated)...")
    negative_pairs = create_mixed_pairs(human.sample(min(len(human), 300)), 
                                       generated.sample(min(len(generated), 300)), 
                                       sample_size//2 if sample_size else 2000, 
                                       label=0)
    data.extend(negative_pairs)
    
    # Create DataFrame and save to CSV
    output_df = pd.DataFrame(data)
    output_df.to_csv(output_file, index=False)
    
    print(f"Created dataset with {len(output_df)} pairs")
    print(f"Class distribution: {output_df['label'].value_counts().to_dict()}")
    print(f"Saved to {output_file}")
    
    return True

def create_pairs_from_group(df, sample_size=None, label=1):
    """Create pairs of essays from the same group (generated or human)"""
    pairs = []
    essays = df['text'].tolist()
    ids = df.index.tolist()
    
    # Get a limited number of pairs to avoid memory issues
    max_pairs = min(sample_size or 1000, 1000)
    
    # Instead of creating all possible pairs at once, we'll sample randomly
    if len(essays) > 1:
        # Number of attempts to generate unique pairs
        attempts = min(max_pairs * 3, len(essays) * (len(essays) - 1) // 2)
        
        print(f"Creating up to {max_pairs} pairs from {len(essays)} essays")
        
        # Generate random pairs without storing all possible combinations
        for _ in tqdm(range(attempts)):
            if len(pairs) >= max_pairs:
                break
                
            # Get random indices
            i = random.randint(0, len(essays) - 1)
            j = random.randint(0, len(essays) - 1)
            
            # Skip if same essay or pair already exists
            if i == j:
                continue
                
            # Get the actual essays
            text1 = essays[i]
            text2 = essays[j]
            
            # Skip if either text is too short
            if len(text1.split()) < 50 or len(text2.split()) < 50:
                continue
            
            # Add pair to dataset
            pairs.append({
                'text1': text1,
                'text2': text2,
                'id1': ids[i],
                'id2': ids[j],
                'label': label
            })
    
    print(f"Created {len(pairs)} pairs with label {label}")
    return pairs

def create_mixed_pairs(df1, df2, sample_size=None, label=0):
    """Create pairs with one essay from each group"""
    pairs = []
    essays1 = df1['text'].tolist()
    essays2 = df2['text'].tolist()
    ids1 = df1.index.tolist()
    ids2 = df2.index.tolist()
    
    # Limit number of pairs for memory efficiency
    max_pairs = min(sample_size or 2000, 2000)
    
    print(f"Creating up to {max_pairs} mixed pairs from {len(essays1)} and {len(essays2)} essays")
    
    # Generate random pairs without storing all possible combinations
    attempts = min(max_pairs * 3, len(essays1) * len(essays2))
    
    for _ in tqdm(range(attempts)):
        if len(pairs) >= max_pairs:
            break
            
        # Get random indices
        i = random.randint(0, len(essays1) - 1)
        j = random.randint(0, len(essays2) - 1)
        
        # Get the actual essays
        text1 = essays1[i]
        text2 = essays2[j]
        
        # Skip if either text is too short
        if len(text1.split()) < 50 or len(text2.split()) < 50:
            continue
        
        # Add pair to dataset
        pairs.append({
            'text1': text1,
            'text2': text2,
            'id1': ids1[i],
            'id2': ids2[j],
            'label': label
        })
    
    print(f"Created {len(pairs)} mixed pairs with label {label}")
    return pairs

def main():
    parser = argparse.ArgumentParser(description='Download and prepare DAIGT dataset for plagiarism detection')
    parser.add_argument('--download', action='store_true', help='Download DAIGT dataset from Kaggle')
    parser.add_argument('--input_dir', type=str, default='dataset', help='Directory containing DAIGT dataset files')
    parser.add_argument('--output_file', type=str, default='dataset/plagiarism_dataset.csv', help='Output CSV file')
    parser.add_argument('--sample_size', type=int, default=5000, help='Number of pairs to generate')
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    
    # Download dataset if requested
    if args.download:
        download_daigt_dataset(args.input_dir)
    
    # Convert dataset to plagiarism format
    convert_to_plagiarism_format(args.input_dir, args.output_file, args.sample_size)

if __name__ == "__main__":
    main() 