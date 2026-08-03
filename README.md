# Semantic Text Summarizer

A modern **extractive text summarization** application built with Sentence Transformers, cosine similarity, graph construction, and PageRank. The project identifies the most important sentences in a document and presents them through an interactive Streamlit interface.

> This project was created as a practical NLP learning project to demonstrate semantic sentence embeddings and graph-based summarization.

## Project Preview

![Extractive Text Summarization Pipeline](assets/pipeline.png)

## Key Features

- Extractive text summarization without generating new sentences.
- Semantic sentence representations using `distiluse-base-multilingual-cased-v2`.
- Sentence-to-sentence comparison using cosine similarity.
- Graph-based sentence ranking with NetworkX PageRank.
- User-controlled summary length from 1 to 15 sentences.
- Original word count, summary word count, and reduction percentage.
- Responsive Streamlit interface with custom external CSS.
- Automatic loading of the required NLTK resources.

## How It Works

```text
User Input Text
      ↓
Sentence Tokenization
      ↓
Basic Text Cleaning
      ↓
Sentence Embeddings
      ↓
Cosine Similarity Matrix
      ↓
Graph Construction
      ↓
PageRank Algorithm
      ↓
Sentence Ranking
      ↓
Select Top Sentences
      ↓
Restore Original Order
      ↓
Display Summary
```

### Pipeline Explanation

1. **Sentence Tokenization** — divides the input document into individual sentences.
2. **Basic Text Cleaning** — converts sentences to lowercase and removes English stopwords.
3. **Sentence Embeddings** — transforms every cleaned sentence into a semantic vector.
4. **Cosine Similarity** — measures the semantic relationship between every pair of sentences.
5. **Graph Construction** — represents sentences as nodes and similarity scores as weighted edges.
6. **PageRank** — calculates an importance score for each sentence.
7. **Sentence Selection** — chooses the highest-ranked sentences requested by the user.
8. **Original-Order Restoration** — places selected sentences back in their original document order.
9. **Summary Display** — presents the final extractive summary and its statistics.

## Technology Stack

| Technology | Purpose |
|---|---|
| Python 3.11 | Core programming language |
| Streamlit | Interactive web interface |
| Sentence Transformers | Semantic sentence embeddings |
| Scikit-learn | Cosine similarity calculation |
| NetworkX | Graph construction and PageRank |
| NLTK | Sentence tokenization and stopword removal |
| NumPy | Similarity matrix operations |
| CSS | Custom interface styling |

## Project Structure

```text
semantic-textrank-summarizer/
│
├── app.py
├── style.css
├── requirements.txt
├── README.md
└── assets/
    └── pipeline.png
```

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/semantic-textrank-summarizer.git
cd semantic-textrank-summarizer
```

Replace `YOUR_USERNAME` with your GitHub username.

### 2. Create a virtual environment

Using `venv`:

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

Activate it on macOS or Linux:

```bash
source .venv/bin/activate
```

### 3. Install the dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the application

```bash
streamlit run app.py
```

The application will open locally at:

```text
http://localhost:8501
```

## How to Use

1. Paste an article or a paragraph containing several sentences.
2. Choose the required number of sentences in the summary.
3. Click **Generate Summary**.
4. Review the selected summary and the reduction statistics.

## Example

### Input

```text
Artificial intelligence is transforming healthcare by enabling faster diagnoses and personalized treatments. Machine learning models can analyze medical images with high accuracy. However, data privacy and bias remain critical challenges. Ethical and responsible AI deployment is essential for patient trust.
```

### Extractive Summary

```text
Artificial intelligence is transforming healthcare by enabling faster diagnoses and personalized treatments. Ethical and responsible AI deployment is essential for patient trust.
```

The exact selected sentences can change depending on the requested summary length and the semantic relationships in the input text.

## Current Limitations

- The application performs **extractive summarization**, so it selects original sentences rather than generating new wording.
- The current cleaning stage uses English stopwords; therefore, the present configuration is best suited for English text.
- Very short input texts may not provide enough sentence relationships for PageRank to produce a meaningful reduction.
- The first application run can take longer because the Sentence Transformer model and NLTK resources may need to be downloaded.

## Possible Future Improvements

- Compare extractive summarization with an abstractive Transformer model.
- Add multilingual preprocessing and language detection.
- Visualize the sentence-similarity graph inside the application.
- Add file upload support for TXT and PDF documents.
- Add ROUGE-based evaluation on a benchmark summarization dataset.
- Add a downloadable summary file.

## Author Note

This project is part of my ongoing learning journey in advanced Natural Language Processing. It focuses on understanding semantic embeddings, cosine similarity, graph-based ranking, and practical NLP deployment with Streamlit.
