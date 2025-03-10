# IPO-Savvy-Chatbot - AI-Powered IPO Stock Analysis Assistant

## 📌 Overview
**IPOSavvy** is an AI-powered platform designed to analyze and provide insights on Initial Public Offerings (IPOs). The system leverages **OpenAI's GPT-4** and **LangChain** for intelligent querying and utilizes **Pinecone** as a vector database for fast and efficient search. The project allows users to explore historical IPO data, understand market trends, and retrieve relevant insights using natural language queries.
In addition to structured IPO data, IPOSavvy also gathers **real-time news articles** from **The Guardian API** . This allows users to access the latest IPO-related news while also performing **sentiment analysis** on article content.

## 🏗️ Tech Stack
- **Programming Language:** Python
- **Backend:** Jupyter Notebook, LangChain, OpenAI API
- **Frontend:** Streamlit
- **Database:** Pinecone (Vector Database)
- **LLM Model:** OpenAI GPT-4

## 📡 Platforms & Libraries Used
- **LangChain**: For building the AI-based retrieval system
- **Pinecone**: For storing and retrieving IPO-related embeddings
- **OpenAI API**: For language model responses
- **Streamlit**: For creating an interactive user interface
- **Pandas**: For handling IPO datasets
- **Requests & BeautifulSoup**: For fetching and scraping news articles
- **NLTK (VADER Sentiment Analysis)**: For analyzing article sentiment

## 📊 Datasets & APIs Used

- **IPO Datasets (CSV format)**:
  - `ipos-2020.csv`
  - `ipos-2021.csv`
  - `ipos-2022.csv`
  - `ipos-2023.csv`
  - `ipos-2024.csv`

**Datasets** gathered from StockAnalysis Website for all IPOs in the last 5 years. 
  
- **APIs Used**:
  - **OpenAI API**: Provides GPT-4 capabilities for answering IPO-related queries.
  - **Pinecone API**: Handles vector storage and fast retrieval for IPO data embeddings.
  - **The Guardian API**: Fetches real-time news articles about IPOs.
  - **Financial Modeling Prep API**: Fetches balance sheet statements for all IPO names  gathered from StockAnalysis data.


## 🛠️ Setup & Installation
### Prerequisites
- Python 3.8+
- OpenAI API Key
- Pinecone API Key
- The Guardian API Key
- FMP API Key

### Installation Steps
1. **Clone the repository**
2. **Install dependencies**
   pip install -r requirements.txt
3. **Setup Environment Variables**
   - export OPENAI_API_KEY=<YOUR_OPENAI_API_KEY>
   - export PINECONE_API_KEY=<YOUR_PINECONE_API_KEY>
   - export GUARDIAN_API_KEY=<YOUR_GUARDIAN_API_KEY>
4. **Run the Streamlit app**
   streamlit run frontendCode.py
