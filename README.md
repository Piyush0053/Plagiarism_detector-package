# Advanced Plagiarism Detection System

A comprehensive web application for detecting plagiarism using traditional text-matching techniques and advanced NLP models including custom-trained neural networks.

## Features

- **Multiple Detection Methods**:
  - Basic text similarity (cosine similarity, Jaccard similarity)
  - Advanced NLP models (Word2Vec, BERT, Sentence Transformers)
  - Custom-trained plagiarism detection model

- **Web Interface**:
  - User-friendly UI for text comparison
  - File upload support
  - Web search for potential plagiarism sources
  - Interactive visualization of results

- **Custom Model Training**:
  - Train your own plagiarism detection model using the DAIGT dataset
  - Customizable model parameters
  - Fine-tunable thresholds for detection

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/plagiarism-detector.git
cd plagiarism-detector
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up Kaggle API credentials (for dataset download):
   - Create a Kaggle account if you don't have one
   - Go to Account > Create API Token to download kaggle.json
   - Place the kaggle.json file in ~/.kaggle/ (Unix/Linux/MacOS) or %USERPROFILE%\.kaggle\ (Windows)

## Usage

### Running the Web Application

```bash
cd web_app
python app.py
```

Then open your browser and navigate to `http://127.0.0.1:5000/`.

### Training a Custom Model

1. Download and prepare the dataset:
```bash
python prepare_daigt_dataset.py --download
```

2. Train the model:
```bash
python web_app/run_training.py --epochs 3 --batch_size 4
```

### Testing the Custom Model

```bash
python test_custom_model.py --text1 "Original text here" --text2 "Potentially plagiarized text here"
```

## Advanced Usage

### Using Different Pre-trained Models

You can select different pre-trained models for the custom training:

```bash
python web_app/run_training.py --model_name "microsoft/deberta-v3-base" --epochs 3
```

Available model options:
- microsoft/deberta-v3-small
- microsoft/deberta-v3-base
- bert-base-uncased
- roberta-base

### Configuring Web Search

The system can search the web for potential plagiarism sources. This feature uses the Requests library to search the web.

## Project Structure

```
plagiarism-detector/
├── web_app/
│   ├── app.py                 # Flask application
│   ├── model_training.py      # Custom model training
│   ├── custom_model_inference.py  # Inference with custom model
│   ├── nlp_utils.py           # NLP utilities
│   ├── model/                 # Directory for trained models
│   └── templates/             # HTML templates
├── dataset/                   # Dataset storage
├── prepare_daigt_dataset.py   # Dataset preparation script 
├── test_custom_model.py       # Script to test the model
├── requirements.txt           # Project dependencies
└── README.md                  # Project documentation
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- DAIGT dataset used for training the custom model
- HuggingFace Transformers library for pre-trained models
- Flask for the web framework 