# YouTube Transcript QA using RAG Pipeline

This project is a simple Retrieval-Augmented Generation (RAG) pipeline that answers questions based on YouTube video transcripts.

We use freely available models from Hugging Face and process the transcript data to make it searchable and answerable using natural language questions.

## 🔍 What It Does

- Takes a YouTube video transcript
- Splits and indexes the transcript into chunks
- Uses a retriever to fetch relevant chunks based on a user query
- Generates a natural language answer using a Hugging Face language model

## 🛠️ Technologies Used

- Python
- Hugging Face Transformers
- Hugging Face Datasets
- FAISS (for vector search)
- Sentence Transformers (for embeddings)
- YouTube Transcript API

## 📦 How It Works (Simple Steps)

1. **Get Transcript**: Automatically fetch or manually load a transcript from a YouTube video.
2. **Preprocess**: Clean and split the transcript into chunks.
3. **Embed**: Convert chunks into vector embeddings using a Sentence Transformer model.
4. **Store**: Save the embeddings in a FAISS vector database.
5. **Ask Questions**: When a question is asked, the pipeline retrieves the most relevant chunks.
6. **Generate Answer**: A Hugging Face model generates a response based on the retrieved information.

## 🚀 Getting Started

### Prerequisites

Install the required packages:

```bash
pip install transformers datasets faiss-cpu sentence-transformers youtube-transcript-api
