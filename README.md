# chat-with-pdf-using-NvidiaNim
using NvidiaNim to chat with pdf document

![Application Screenshot](https://github.com/leodeveloper/chat-with-pdf-using-NvidiaNim/blob/main/Nvidia%20Nim.png)

# Chat with multiple PDF file using NvidiaNim

This project allows you to upload multiple PDF files and interact with them using natural language questions and answers. It leverages NVIDIAEmbeddings, ChatNVIDIA, OllamaEmbeddings, and FAISS to provide efficient and accurate responses.

## Features

- Upload and process multiple PDF documents.
- Use NVIDIA's powerful embeddings for natural language understanding.
- Efficient querying and retrieval with FAISS.
- Simple and intuitive user interface with Streamlit.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/leodeveloper/chat-with-pdf-using-NvidiaNim.git
   cd chat-with-pdf-using-NvidiaNim

   Set up environment variables by copying .env.example to .env and filling in the necessary values.


   pip install -r requirements.txt
   streamlit run app.py


How It Works
NVIDIAEmbeddings: Utilized for generating embeddings from the text within the PDFs.
ChatNVIDIA: Handles the natural language processing and interaction.
OllamaEmbeddings: Provides additional embedding capabilities to enhance understanding.
FAISS: Efficiently indexes and searches through embeddings for quick responses.
Contributing
Contributions are welcome! Please fork the repository and create a pull request with your changes.

## feel free contact on this email leodeveloper@gmail.com