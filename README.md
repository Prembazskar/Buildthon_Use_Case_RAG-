🌾 Tamil Nadu Agriculture Schemes RAG Assistant

An AI-powered Retrieval-Augmented Generation (RAG) application that helps farmers find information about Tamil Nadu agriculture schemes using indexed PDF documents and linked web pages.

The application uses LangChain, FAISS, OpenAI, and Streamlit to retrieve relevant information and generate clear, evidence-based answers in English, with source references.

🚀 Features

PDF ingestion: Extracts text from multiple agriculture scheme PDFs.

Web content ingestion: Discovers and extracts text from supported URLs found in PDFs.

Vector database: Uses FAISS for efficient semantic similarity search.

AI-powered answers: Uses OpenAI to generate answers based on retrieved documents.

English responses: Translates relevant Tamil content into clear English.

Source references: Displays source filenames, PDF page numbers, URLs, and retrieved text.

Chat history: Keeps previous questions and answers visible during the current Streamlit session.

Interactive dashboard: Provides a Streamlit chat interface with an optional agriculture banner.

Knowledge-base-based responses: Designed to answer from indexed content rather than performing live web searches during questions.

