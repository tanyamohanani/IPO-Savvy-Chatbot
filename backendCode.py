#RUN THIS FILE AS .ipynb FILE
#Import libraries

!pip install langchain_pinecone
!pip install openai
!pip install langchain
!pip install pinecone-client
!pip install langchain_openai
import openai
import langchain
import pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain_pinecone import Pinecone

#Define API Keys and Pinecone Database Index

import os
os.environ['OPENAI_API_KEY'] = 'Enter Your Key'
os.environ['PINECONE_API_KEY'] = 'Enter Your Key'

index_name = "ipo-analysis"
embeddings = OpenAIEmbeddings()

#Import Stock Analysis Company Data

import pandas as pd

ipo_2020 = pd.read_csv('ipos-2020.csv')
ipo_2021 = pd.read_csv('ipos-2021.csv')
ipo_2022 = pd.read_csv('ipos-2022.csv')
ipo_2023 = pd.read_csv('ipos-2023.csv')
ipo_2024 = pd.read_csv('ipos-2024.csv')

df = pd.concat([ipo_2020,ipo_2021,ipo_2022,ipo_2023,ipo_2024]).sort_values(by=['IPO Date'],ascending=False,ignore_index=True)

df.head()

#Convert Stock Analysis data from CSV to text

df['IPO Date']=df['IPO Date'].astype(str)
df['IPO Price']=df['IPO Price'].astype(str)
df['Current']=df['Current'].astype(str)
df['Market Cap']=df['Market Cap'].astype(str)
df['Open Price']=df['Open Price'].astype(str)
df['Revenue']=df['Revenue'].astype(str)
df['EBITDA']=df['EBITDA'].astype(str)
df['Investing CF']=df['Investing CF'].astype(str)
df['Financing CF']=df['Financing CF'].astype(str)

cols_to_convert = ['IPO Price', 'Current', 'Market Cap', 'Open Price', 'Revenue', 'EBITDA', 'Investing CF', 'Financing CF']
for col in cols_to_convert:
    df[col] = df[col].astype(str).replace('nan', 'is not specified')

# Create the text for embedding
text_embedding = (df["Company Name"] + " with the Symbol: " + df['Symbol'] +
                            " has an IPO date of " + df['IPO Date'] +
                            " with the IPO Price " + df['IPO Price'] +
                            " with a current stock price of " + df['Current'] +
                            " and a market cap of " + df['Market Cap'] +
                            " and open price " + df['Open Price'] +
                            " and revenue " + df['Revenue'] +
                            " and EBITDA " + df['EBITDA'] +
                            " and investing cash flows (CF) " + df['Investing CF'] +
                            " and financing cash flows (CF) " + df['Financing CF'] + ".")

# Convert dataframe text_embedding to .txt file
with open("ipo_data.txt", "w") as file:
     file.write("\n".join(text_embedding))

# Load stock analysis data into Textloader and create chunks of data

loader = TextLoader("ipo_data.txt")       # FMP API text file (balance sheets)
documents = loader.load()
def chunk_data(docs,chunk_size=1500,chunk_overlap=300):
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=chunk_size,chunk_overlap=chunk_overlap)
    doc=text_splitter.split_documents(docs)
    return doc
documents=chunk_data(docs=documents)
len(documents)

# Use Pinecone Vector Store to embed data and store it in Pinecone DB

vectorstore_from_docs = PineconeVectorStore.from_documents(
        documents,
        index_name=index_name,
        embedding=embeddings,

    )

# Retrieve balance sheets from FMP API and convert them to text

#!/usr/bin/env python

import urllib.parse
import ssl
import time
from urllib.error import HTTPError

# Batch 1
companies_batch1 = df['Symbol'][1663:1913]
company_names_batch1 = df['Company Name'][1663:1913]
api_key_batch1 = "OqGLfcx3sfOxktgm604zy05GzCN2JKU2"

# Batch 2
companies_batch2 = df['Symbol'][1413:1663]
company_names_batch2 = df['Company Name'][1413:1663]
api_key_batch2 = "yYnfZXylioKWNojdVIz4g4WB0UHxd2mr"

# Batch 3
companies_batch3 = df['Symbol'][1163:1413]
company_names_batch3 = df['Company Name'][1163:1413]
api_key_batch3 = "2yOjAn0KHbE0D82Mu3iKeovZP2baXoim "

# Batch 4
companies_batch4 = df['Symbol'][913:1163]
company_names_batch4 = df['Company Name'][913:1163]
api_key_batch4 = "EWVi1QVfnvl8u9zbycYEG7oBsPmNMPrC"

# Batch 5
companies_batch5 = df['Symbol'][663:913]
company_names_batch5 = df['Company Name'][663:913]
api_key_batch5 = "CzEFE4tBhVDs0u9aIz76Gr02MSN4Ddjo"

# Batch 6
companies_batch6 = df['Symbol'][413:663]
company_names_batch6 = df['Company Name'][413:663]
api_key_batch6 = "zFaxBTDPHMtpz341HQsE2SUJW7AHUMBQ"

# Batch 7
companies_batch7 = df['Symbol'][163:413]
company_names_batch7 = df['Company Name'][163:413]
api_key_batch7 = "k8QHC7WSIW5aWyCzl5R4w7QoiJSYOiRv"

# Batch 8
companies_batch8 = df['Symbol'][0:163]
company_names_batch8 = df['Company Name'][0:163]
api_key_batch8 = "oeIF8D0C2B0rcOw20KpyuyT6X6QYuRhX"

text1 = []
#text2=[]

max_retries = 5
retry_delay = 1  # Initial retry delay in seconds

def get_text_data(url):
    try:
        # For Python 3.0 and later
        from urllib.request import urlopen
    except ImportError:
        # Fall back to Python 2's urllib2
        from urllib2 import urlopen

    import certifi

    context = ssl.create_default_context(cafile=certifi.where())
    with urlopen(url, context=context) as response:
        data = response.read().decode("utf-8")
        return data

def process_batch(companies, company_names, api_key, text_data_var):
    for index, (company, company_name) in enumerate(zip(companies, company_names)):
        encoded_company = urllib.parse.quote(company)
        url = f'https://financialmodelingprep.com/api/v3/balance-sheet-statement/{encoded_company}?period=annual&apikey={api_key}'
        print("URL:", url)

        retry_count = 0
        while retry_count < max_retries:
            try:
                text_data = get_text_data(url)
                print("Text Data:", text_data)
                text_data_var = company_name + " " + text_data
                text1.append(text_data_var)
                break  # Break out of the retry loop if the request is successful
            except HTTPError as e:
                if e.code == 429:
                    print(f"Rate limit exceeded. Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Double the retry delay for the next attempt
                    retry_count += 1
                else:
                    print(f"Error occurred: {str(e)}")
                    break  # Break out of the retry loop for other types of errors
            except Exception as e:
                print(f"Error occurred: {str(e)}")
                break  # Break out of the retry loop for other types of errors

        if retry_count == max_retries:
            print(f"Max retries reached. Skipping {company}.")

# Process Batch 1
process_batch(companies_batch1, company_names_batch1, api_key_batch1, "text_data1")

# Process Batch 2
process_batch(companies_batch2, company_names_batch2, api_key_batch2, "text_data2")

# Process Batch 3
process_batch(companies_batch3, company_names_batch3, api_key_batch3, "text_data3")

# Process Batch 4
process_batch(companies_batch4, company_names_batch4, api_key_batch4, "text_data4")

# Process Batch 5
process_batch(companies_batch5, company_names_batch5, api_key_batch5, "text_data5")

# Process Batch 6
process_batch(companies_batch6, company_names_batch6, api_key_batch6, "text_data6")

# Process Batch 7
process_batch(companies_batch7, company_names_batch7, api_key_batch7, "text_data7")

# Process Batch 8
process_batch(companies_batch8, company_names_batch8, api_key_batch8, "text_data8")

# Save text1 as a .txt file
with open("balancesheets.txt", "w") as file:
     file.write("\n".join(text2))

# Load FMP data (balance sheet statements) into Textloader and create chunks of data

loader = TextLoader("balancesheets.txt")       # FMP API text file (balance sheets)
documents = loader.load()
def chunk_data(docs,chunk_size=1500,chunk_overlap=300):
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=chunk_size,chunk_overlap=chunk_overlap)
    doc=text_splitter.split_documents(docs)
    return doc
documents=chunk_data(docs=documents)
len(documents)

# Use Pinecone Vector Store to embed data and store it in Pinecone DB

vectorstore_from_docs = PineconeVectorStore.from_documents(
        documents,
        index_name=index_name,
        embedding=embeddings,

    )

# Retrive new articles using APIs

!pip install beautifulsoup4
from bs4 import BeautifulSoup


# Initialize NLTK's VADER sentiment analyzer
sid = SentimentIntensityAnalyzer()


api_key = 'c19fb403-870c-4605-93e4-999a02628c49'

# companies = ipo_all['Company Name'][1312:1912]
# companies = ipo_all['Company Name'][1312:1812]
#companies = ipo_all['Company Name'][812:1312]
companies = ipo_all['Company Name'][312:812]


# Specify the number of days for the time period
num_days = 7

# Calculate dates for the specified time period
to_date = datetime.now()
from_date = to_date - timedelta(days=num_days)

# Format dates for API request
to_date_str = to_date.strftime('%Y-%m-%d')
from_date_str = from_date.strftime('%Y-%m-%d')

# Base URL for the NewsAPI
#url_base = f'https://newsapi.org/v2/everything?apiKey={api_key}'
combined_data_1 = {}
#combined_data = {}
total_articles = 0  # Counter to track the total number of articles retrieved


# This will store all content from all articles
all_content = []
#num_articles = 0;
for company in companies:
    # Update the request URL with the current company in the query and date range

    url = f'https://content.guardianapis.com/search?q={company}&from-date={from_date_str}&to-date={to_date_str}&api-key={api_key}'

    # Making the API call
    response = requests.get(url)

    if response.status_code == 200:  # Checking if the request was successful
        #num_articles += response.json().get('total')
        #articles = response.json().get('articles', []) -- newsapi
        #articles = response.json().get('response', {}).get('docs', []) --nyt api
        articles = response.json().get('response', {}).get('results', []) #-- guardian api
        #print(response.json())
        company_data = []

        for article in articles:
            #content = article.get('body') # -- guardian api
            #content = article.get('snippet') -- nyt api
            #content = article.get('content', '') -- newsapi

            #if content:
                #total_articles += 1
            web_url = article.get('webUrl')  # Get the URL of the article's webpage
            if web_url:
                # Use BeautifulSoup for web scraping to extract article content from webpage
                article_response = requests.get(web_url)
                if article_response.status_code == 200:
                    # Parse HTML content using BeautifulSoup
                    soup = BeautifulSoup(article_response.text, 'html.parser')
                    # Find the element containing the article content
                    content_element = soup.find('div', attrs={'data-gu-name': 'body'})
                    if content_element:
                        # Extract text content from the element
                        content = content_element.get_text(separator='\n')
                        # Perform sentiment analysis using NLTK's VADER
                        sentiment = sid.polarity_scores(content)
                        sentiment_summary = f"Polarity: {sentiment['compound']:.2f}"
                        # Combine content with sentiment analysis
                        combined_content = f"Content: {content}\nSentiment: {sentiment_summary}"
                        company_data.append(combined_content)
                        total_articles += 1
                    else:
                        print(f"Failed to extract content from article: {web_url}")
                else:
                    print(f"Failed to fetch article webpage: {web_url}")
                # Combine content with sentiment analysis
                #combined_content = f"Content: {content}\nSentiment: {sentiment_summary}"
                #company_data.append(combined_content)
            else:
                print(f"Failed to fetch articles for {company}")
        combined_data_1[company] = company_data
    else:
        print(f"Error fetching data for {company}: {response.status_code}")


print(f"Total articles retrieved: {total_articles}")
for company, data in combined_data_1.items():
    print(f"\nCompany: {company}\n{'='*20}")
    for entry in data:
        print(entry, "\n")


# Convert all articles to a list of strings

iter_2 = [f"{key} {value}" for key, value in combined_data_1.items()] # next 500 companies + 100 + 500

# Split into chunks and embed data

import openai

# Function to split text into chunks with overlap
def chunk_text(text, chunk_size=7500, chunk_overlap=100):
    chunks = []
    start = 0
    while start < len(text):
        if start > 0:
            start -= chunk_overlap
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - chunk_overlap
        if start >= len(text):
            break
    return chunks

# Function to process a list of documents, splitting each into chunks
def chunk_data(docs, chunk_size=7500, chunk_overlap=100):
    all_chunks = []
    for doc in docs:
        chunks = chunk_text(doc, chunk_size, chunk_overlap)
        all_chunks.extend(chunks)
    return all_chunks

# Function to generate embeddings for a list of text chunks using OpenAI's API
def generate_embeddings(text_chunks, model="text-embedding-3-small"):
    openai.api_key = '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'  # Replace with your actual API key
    embeddings = []
    for chunk in text_chunks:
        try:
            response = openai.Embedding.create(
                input=chunk,
                model=model
            )
            embeddings.append(response['data'][0]['embedding'])
        except Exception as e:
            print("Error processing chunk:", chunk)
            print(e)
    return embeddings


chunks = chunk_data(iter_2, chunk_size=7500, chunk_overlap=100)
#embeddings = generate_embeddings(chunks)
#embeddings_1 = generate_embeddings(chunks)
#embeddings_2 = generate_embeddings(chunks)
embeddings_3 = generate_embeddings(chunks)
# Print the first few embeddings to check (optional)
for embedding in embeddings_3[:1]:  # Show first 5 embeddings
    print(embedding)

# Insert embeddings into pinecone

import pinecone
from pinecone import Pinecone

pc = Pinecone(api_key='d677dc0b-e405-4d3c-8a89-afd69503e8fd')
index = pc.Index("ipo-analysis")

def upload_embeddings_to_pinecone(embeddings):
    # Generate unique IDs for each embedding
    ids = [f"embedding-{i+9971}" for i in range(len(embeddings))]
    # Prepare the data for Pinecone
    vectors = list(zip(ids, embeddings))

    # Insert embeddings in batches of 100
    batch_size = 100
    for i in range(0, len(vectors), batch_size):
        batch = vectors[i:i + batch_size]
        index.upsert(vectors=batch)
        
upload_embeddings_to_pinecone(embeddings_3)




