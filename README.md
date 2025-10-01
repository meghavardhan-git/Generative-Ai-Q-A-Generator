# Generative AI Q&A Generator

A Flask-based application that generates **Question & Answer pairs** from input text using **Generative AI**.  
It leverages **LangChain**, **ChromaDB**, and **Gemini/OpenAI** models to process text and return meaningful Q&A pairs.

---

## 🚀 Features
- Generate Q&A pairs from any given text or document
- Backend powered by Flask and LangChain
- Vector storage and retrieval with **ChromaDB**
- Supports **Gemini** (Google Generative AI) and **OpenAI LLMs**
- Simple UI for user interaction

---

## 🧱 Tech Stack
- Python 3.10
- Flask
- LangChain
- ChromaDB
- Gemini / OpenAI API

---

## 🔧 Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/meghavardhan-git/Generative-Ai-Q-A-Generator.git
cd Generative-Ai-Q-A-Generator
```
### 2. Create and activate Conda environment
```bash
conda create -n llmapp python=3.10 -y
conda activate llmapp
```
### 3.Install dependencies
pip install -r requirements.txt

### 4.Configure API Keys

Create a .env file in the root directory and add your keys:

GOOGLE_API_KEY="your_gemini_api_key_here"

### 5.Run the application
python app.py

Open local host in your Browser

