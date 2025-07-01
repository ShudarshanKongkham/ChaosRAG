# ChaosRAG - Landmine Identification System

A Retrieval-Augmented Generation (RAG) system designed to assist defense personnel in identifying landmines based on field descriptions and intelligence data.

## Features

- **Expert EOD Analysis**: Acts as a specialized explosive ordnance disposal expert
- **Landmine Identification**: Identifies landmine types based on physical descriptions
- **Safety Protocols**: Emphasizes safety considerations and proper EOD procedures
- **Source Verification**: Provides reference sources for all identifications
- **Interactive Interface**: User-friendly command-line interface with help system

## Technology Stack

- **LangChain**: Framework for building LLM applications
- **Ollama**: Local LLM inference (LLaMA 3.1)
- **FAISS**: Vector database for document retrieval
- **Sentence Transformers**: Text embeddings
- **PyPDF**: PDF document processing

## Installation

1. Clone the repository:
```bash
git clone https://github.com/YOUR_USERNAME/ChaosRAG.git
cd ChaosRAG
```

2. Install required packages:
```bash
pip install langchain langchain-community sentence-transformers faiss-gpu pypdf tf-keras
```

3. Install Ollama and download LLaMA 3.1:
```bash
# Install Ollama from https://ollama.ai
ollama pull llama3.1
ollama pull nomic-embed-text
```

## Usage

1. Place your PDF documents in the project directory
2. Run the Jupyter notebook: `RAG_backbone.ipynb`
3. Follow the interactive prompts to describe suspected landmines
4. Receive expert analysis and safety recommendations

## Safety Notice

⚠️ **CRITICAL**: This system is for educational and training purposes. Always follow proper EOD protocols and contact qualified specialists for real explosive ordnance situations.

## Contributing

Contributions are welcome! Please read our contributing guidelines and submit pull requests for any improvements.

## License
This project is licensed under the MIT License 