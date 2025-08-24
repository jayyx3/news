# Overview

This is a QA Tool for French News Transitions - a quality assurance engine that analyzes French news article transition phrases. The application evaluates whether short French transition phrases between news paragraphs meet specific quality criteria including thematic cohesion, word count limits, and placement rules. The system uses NLP techniques to assess semantic similarity and detect lemma repetition across articles, providing clear, explainable results through a Streamlit interface.

# User Preferences

Preferred communication style: Simple, everyday language.

# System Architecture

## Frontend Architecture
- **Streamlit Web Application**: Interactive interface for uploading articles, configuring analysis parameters, and displaying results
- **Configuration Controls**: Sidebar with adjustable similarity thresholds for transition analysis
- **Results Display**: Tabular format showing analysis outcomes with downloadable CSV/HTML exports
- **Real-time Processing**: Cached resource loading for NLP models to improve performance

## Backend Architecture
- **Modular Design**: Separated concerns across multiple Python modules:
  - `qa_engine.py`: Core analysis engine with French NLP capabilities
  - `text_processor.py`: Text parsing and structure extraction
  - `utils.py`: Helper functions for file parsing and data transformation
  - `app.py`: Streamlit application entry point

## NLP Processing Pipeline
- **French Language Support**: Primary use of spaCy French models (`fr_core_news_md` with `fr_core_news_sm` fallback)
- **Semantic Analysis**: Sentence transformers for multilingual semantic similarity (`paraphrase-multilingual-MiniLM-L12-v2` with `distiluse-base-multilingual-cased` fallback)
- **Text Processing**: Paragraph segmentation, transition identification, and lemmatization for repetition detection

## Quality Assessment Rules
- **Rule-based Validation**: Deterministic checks for word count limits (≤5 words) and concluding transition placement
- **Semantic Cohesion Analysis**: Dual similarity scoring system:
  - Transition-to-next paragraph similarity (should be high)
  - Transition-to-previous paragraph similarity (should be low)
- **Repetition Detection**: Article-wide lemma/root repetition tracking using French linguistic models

## Data Processing
- **Article Structure**: Parsed input format with title, subtitle, article content, and generated transitions
- **Structured Output**: Comprehensive analysis results including pass/fail status, failure reasons, triggered rules, and similarity scores
- **Export Capabilities**: CSV and HTML download functionality for analysis results

# External Dependencies

## Core Python Libraries
- **streamlit**: Web application framework for the user interface
- **pandas**: Data manipulation and analysis for results processing
- **plotly**: Interactive visualizations and charts (express and graph_objects)
- **numpy**: Numerical computing for similarity calculations

## NLP and Machine Learning
- **spaCy**: French natural language processing pipeline (`fr_core_news_md`, `fr_core_news_sm`)
- **sentence-transformers**: Multilingual semantic similarity models
  - Primary: `paraphrase-multilingual-MiniLM-L12-v2`
  - Fallback: `distiluse-base-multilingual-cased`

## Text Processing
- **re**: Regular expressions for text parsing and pattern matching
- **io.StringIO**: String buffer operations for file handling
- **time**: Performance monitoring and caching utilities

## Development Environment
- **Replit Compatibility**: Designed to run on both local environments and Replit platform
- **Model Fallbacks**: Graceful degradation when preferred NLP models are unavailable
- **Error Handling**: Comprehensive exception management for model loading and text processing failures