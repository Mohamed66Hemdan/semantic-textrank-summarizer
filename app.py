from pathlib import Path
from textwrap import dedent
import html

import streamlit as st
import pandas as pd
import numpy as np
import nltk
import networkx as nx

from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer


# =========================================================
# Page Configuration
# =========================================================
st.set_page_config(
    page_title="Semantic Text Summarizer",
    page_icon="📝",
    layout="centered",
    initial_sidebar_state="collapsed",
)


# =========================================================
# Paths
# =========================================================
BASE_DIR = Path(__file__).resolve().parent


# =========================================================
# Load External CSS
# =========================================================
def load_css(css_file: str) -> None:
    css_path = BASE_DIR / css_file

    if not css_path.exists():
        st.error(f"CSS file not found: {css_path}")
        st.stop()

    css_content = css_path.read_text(encoding="utf-8")
    st.markdown(f"<style>{css_content}</style>", unsafe_allow_html=True)


# =========================================================
# Render HTML Safely
# =========================================================
def render_html(content: str) -> None:
    clean_content = dedent(content).strip()

    # st.html is preferred because it renders raw HTML directly.
    if hasattr(st, "html"):
        st.html(clean_content)
    else:
        # Fallback for older Streamlit versions.
        st.markdown(clean_content, unsafe_allow_html=True)


load_css("style.css")


# =========================================================
# Download NLTK Resources
# =========================================================
@st.cache_resource
def download_nltk_resources() -> None:
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt")

    # Some recent NLTK versions also require punkt_tab.
    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        try:
            nltk.download("punkt_tab")
        except Exception:
            pass

    try:
        nltk.data.find("corpora/stopwords")
    except LookupError:
        nltk.download("stopwords")


download_nltk_resources()


# =========================================================
# Load Sentence Transformer Model
# =========================================================
@st.cache_resource(show_spinner=False)
def load_embedding() -> SentenceTransformer:
    return SentenceTransformer("distiluse-base-multilingual-cased-v2")


# =========================================================
# Hero Section
# =========================================================
render_html(
    """
    <div class="hero-container">
        <div class="hero-badge">NLP PROJECT</div>

        <div class="hero-icon">📝</div>

        <h1 class="hero-title">Semantic Text Summarizer</h1>

        <p class="hero-subtitle">
            Generate concise extractive summaries using semantic sentence
            embeddings, cosine similarity, graph construction, and the
            TextRank ranking algorithm.
        </p>

        <div class="tags-container">
            <span class="technology-tag">Sentence Transformers</span>
            <span class="technology-tag">Semantic Embeddings</span>
            <span class="technology-tag">Cosine Similarity</span>
            <span class="technology-tag">PageRank</span>
            <span class="technology-tag">Streamlit</span>
        </div>
    </div>
    """
)


# =========================================================
# Input Section Header
# =========================================================
render_html(
    """
    <div class="section-header">
        <div class="section-icon">01</div>

        <div>
            <h2 class="section-title">Enter your text</h2>

            <p class="section-description">
                Paste an article or paragraph, then choose how many
                sentences you want in the generated summary.
            </p>
        </div>
    </div>
    """
)


# =========================================================
# Text Input
# =========================================================
text = st.text_area(
    label="Input text",
    label_visibility="collapsed",
    placeholder=(
        "Paste your text here...\n\n"
        "For better results, enter a paragraph containing several sentences."
    ),
    height=240,
)


# =========================================================
# Summary Length Input
# =========================================================
number_of_summary = st.number_input(
    label="Number of sentences in the summary",
    min_value=1,
    max_value=15,
    value=5,
    step=1,
)


# =========================================================
# Generate Summary Button
# =========================================================
summarize_button = st.button(
    "✨ Generate Summary",
    use_container_width=True,
)


# =========================================================
# Original Summarization Pipeline
# =========================================================
if summarize_button:
    if not text.strip():
        st.warning("Please enter some text before generating the summary.")

    else:
        with st.spinner(
            "Analyzing sentence relationships and generating the summary..."
        ):
            # 1. Sentence Tokenization
            sentences = sent_tokenize(text)

            if not sentences:
                st.error("The entered text could not be divided into sentences.")
                st.stop()

            if len(sentences) < number_of_summary:
                st.warning(
                    "The requested summary length is greater than the number "
                    "of sentences in the original text. All available "
                    "sentences will be used."
                )
                number_of_summary = len(sentences)

            # 2. Basic Text Cleaning
            clean_sentencs = [sentence.lower() for sentence in sentences]
            stop_words = stopwords.words("english")

            def remove_stopwords(sentence: str) -> str:
                return " ".join(
                    [word for word in sentence.split() if word not in stop_words]
                )

            cleaned_text = [
                remove_stopwords(sentence) for sentence in clean_sentencs
            ]

            # 3. Sentence Embeddings
            model = load_embedding()
            sentences_vectors = model.encode(cleaned_text)

            # 4. Cosine Similarity Matrix
            similarity_matrix = np.zeros(
                [len(sentences), len(sentences)]
            )

            for i in range(len(sentences)):
                for j in range(len(sentences)):
                    if i != j:
                        similarity_matrix[i][j] = cosine_similarity(
                            sentences_vectors[i].reshape(1, -1),
                            sentences_vectors[j].reshape(1, -1),
                        )[0, 0]

            # 5. Graph Construction
            nx_graph = nx.from_numpy_array(similarity_matrix)

            # 6. PageRank
            scores = nx.pagerank(nx_graph)

            # 7. Sentence Ranking
            ranked_sentences = sorted(
                (
                    (scores[i], sentence, i)
                    for i, sentence in enumerate(sentences)
                ),
                reverse=True,
            )

            # 8. Select Top Sentences
            # 9. Restore Original Order
            summary_sentences = sorted(
                ranked_sentences[:number_of_summary],
                key=lambda item: item[2],
            )

            # 10. Generate Final Summary
            summary_text = ""

            for _, sentence, _ in summary_sentences:
                summary_text += sentence + " "

            summary_text = summary_text.strip()

        # =====================================================
        # Summary Statistics
        # =====================================================
        original_words = len(text.split())
        summary_words = len(summary_text.split())

        if original_words > 0:
            compression_ratio = round(
                (1 - summary_words / original_words) * 100,
                1,
            )
        else:
            compression_ratio = 0

        render_html(
            """
            <div class="result-section-header">
                <div class="section-icon result-number">02</div>

                <div>
                    <h2 class="section-title">Summary results</h2>

                    <p class="section-description">
                        The most important sentences selected by the
                        semantic TextRank pipeline.
                    </p>
                </div>
            </div>
            """
        )

        metric_1, metric_2, metric_3 = st.columns(3)

        with metric_1:
            st.metric(label="Original Words", value=original_words)

        with metric_2:
            st.metric(label="Summary Words", value=summary_words)

        with metric_3:
            st.metric(label="Reduction", value=f"{compression_ratio}%")

        safe_summary = html.escape(summary_text)

        render_html(
            f"""
            <div class="summary-card">
                <div class="summary-card-top">
                    <div class="summary-status-icon">✓</div>

                    <div>
                        <div class="summary-label">Generated Summary</div>

                        <div class="summary-meta">
                            {number_of_summary} selected sentences
                        </div>
                    </div>
                </div>

                <div class="summary-divider"></div>

                <div class="summary-text">{safe_summary}</div>
            </div>
            """
        )


# =========================================================
# Pipeline Information
# =========================================================
render_html(
    """
    <div class="pipeline-card">
        <div class="pipeline-title">How the summarizer works</div>

        <div class="pipeline-steps">
            <div class="pipeline-step">
                <span class="pipeline-number">1</span>
                <span>Tokenization</span>
            </div>

            <span class="pipeline-arrow">→</span>

            <div class="pipeline-step">
                <span class="pipeline-number">2</span>
                <span>Embeddings</span>
            </div>

            <span class="pipeline-arrow">→</span>

            <div class="pipeline-step">
                <span class="pipeline-number">3</span>
                <span>Similarity</span>
            </div>

            <span class="pipeline-arrow">→</span>

            <div class="pipeline-step">
                <span class="pipeline-number">4</span>
                <span>PageRank</span>
            </div>

            <span class="pipeline-arrow">→</span>

            <div class="pipeline-step">
                <span class="pipeline-number">5</span>
                <span>Summary</span>
            </div>
        </div>
    </div>
    """
)


# =========================================================
# Footer
# =========================================================
render_html(
    """
    <div class="project-footer">
        <div class="footer-line"></div>

        <p>
            Extractive summarization powered by
            <strong>Sentence Transformers</strong> and
            <strong>TextRank</strong>.
        </p>
    </div>
    """
)
