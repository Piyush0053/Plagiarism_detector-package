import subprocess
import os
import sys

def extract_texts_from_file(file_path):
    """Extract original and plagiarized texts from the test file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Split by sections
    sections = content.split('#')
    
    # Extract relevant texts
    texts = {}
    for section in sections:
        if not section.strip():
            continue
        
        lines = section.strip().split('\n')
        if len(lines) < 2:
            continue
        
        title = lines[0].strip()
        text = '\n'.join(lines[1:]).strip()
        texts[title] = text
    
    return texts

def run_custom_model_test(text1, text2, threshold=0.5):
    """Run the custom model test on two texts"""
    cmd = [
        sys.executable,
        "test_custom_model.py",
        "--text1", text1,
        "--text2", text2,
        "--threshold", str(threshold)
    ]
    
    print("\n" + "="*80)
    print(f"Running test with threshold: {threshold}")
    print(f"Text 1 (first 100 chars): {text1[:100]}...")
    print(f"Text 2 (first 100 chars): {text2[:100]}...")
    print("="*80 + "\n")
    
    process = subprocess.run(cmd, capture_output=True, text=True)
    print(process.stdout)
    if process.stderr:
        print("Errors:")
        print(process.stderr)

def main():
    """Main function to run tests"""
    print("Running tests for plagiarism detection system")
    
    # Extract texts
    test_file = "test_samples.txt"
    if not os.path.exists(test_file):
        print(f"Error: Test file '{test_file}' not found")
        return
    
    texts = extract_texts_from_file(test_file)
    
    # Test case 1: Paraphrased content - should detect similarity but not 100%
    if "Original Text" in texts and "Plagiarized Version (with modifications)" in texts:
        original = texts["Original Text"]
        paraphrased = texts["Plagiarized Version (with modifications)"]
        run_custom_model_test(original, paraphrased, threshold=0.5)
    
    # Test case 2: Direct copy - should detect very high similarity
    if "Original Academic Paragraph" in texts and "Complete Copy (Plagiarism)" in texts:
        original = texts["Original Academic Paragraph"]
        copied = texts["Complete Copy (Plagiarism)"]
        run_custom_model_test(original, copied, threshold=0.5)
    
    # Test case 3: Different texts - should not detect plagiarism
    if "Original Text" in texts and "Original Academic Paragraph" in texts:
        text1 = texts["Original Text"]
        text2 = texts["Original Academic Paragraph"]
        run_custom_model_test(text1, text2, threshold=0.5)

if __name__ == "__main__":
    main() 