import streamlit as st
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA
from langchain_pinecone import PineconeVectorStore
from langchain_core.messages import AIMessage, HumanMessage
import os

# Set up environment variables for OpenAI and Pinecone API keys
os.environ['OPENAI_API_KEY'] = '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'
os.environ['PINECONE_API_KEY'] = 'd677dc0b-e405-4d3c-8a89-afd69503e8fd'

# Specify the name of the Pinecone index for storing vectors
index_name = "ipo-analysis"

# Initialize OpenAI embeddings for semantic similarity search
embeddings = OpenAIEmbeddings()

# Streamlit page configuration
st.set_page_config(page_title="IPOsavvy AI", page_icon="🤖")
st.title("IPOsavvy AI")

# Configure GPT-4 with low temperature for deterministic outputs and limited tokens for concise responses
llm = ChatOpenAI(
    openai_api_key=os.getenv('OPENAI_API_KEY'),
    model_name='gpt-4',
    temperature=0,  # Deterministic output
    max_tokens=150  # Limit the response length
)

# Initialize Pinecone vector store using the specified index name and embedding setup
vectorstore = PineconeVectorStore(index_name=index_name, embedding=embeddings)

# Create a RetrievalQA chain using the GPT-4 model and Pinecone as the retriever
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",  # Approach to document retrieval and integration
    retriever=vectorstore.as_retriever()
)

# Define a function to handle the chat response generation process
def get_response(user_query, chat_history):
    """
    Generate a response based on the user query and conversation history.
    
    :param user_query: User's latest message or query
    :param chat_history: Previous chat messages exchanged
    :return: Assistant's response content
    """
    # Combine previous chat messages into a readable string format
    chat_history_str = "\n".join([f"{message.type}: {message.content}" for message in chat_history])

    # Formulate an initial prompt using the chat history and user query
    initial_prompt = f"Chat History:\n{chat_history_str}\n\nUser: {user_query}\nAssistant:" 

    # Retrieve relevant documents from the vector store using semantic similarity search
    vector_store_result = vectorstore.similarity_search(user_query, k=6)
    vector_store_result_str = "\n".join([doc.page_content for doc in vector_store_result])

    # Combine all relevant information for a final prompt to be sent to the LLM
    combined_prompt = f"Chat History:\n{chat_history_str}\n\nUser Query: {user_query}\n\nRelevant Information from Vector Store:\n{vector_store_result_str}\n\nFinal Assistant Response:"

    # Construct a message object containing the combined prompt
    human_message = HumanMessage(content=combined_prompt)

    # Retrieve a response from the LLM based on the combined prompt
    final_response = llm([human_message])

    # Return the final response content
    return final_response.content

# Initialize chat history in session state if not already done
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="🚀 Welcome to IPOsavvy AI! 💡 I'm your go-to expert for IPO stock analysis. How can I help you?"),
    ]

# Display the conversation so far (AI and human messages)
for message in st.session_state.chat_history:
    if isinstance(message, AIMessage):
        with st.chat_message("AI"):
            st.write(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.write(message.content)

# Accept new user input via chat input widget
user_query = st.chat_input("Type your message here...")

# Process the user's message if provided
if user_query is not None and user_query != "":
    # Append the user's message to the chat history
    user_query = str(user_query)  # Ensure user query is a string
    st.session_state.chat_history.append(HumanMessage(content=user_query))

    # Display the user's input message in the chat interface
    with st.chat_message("Human"):
        st.markdown(user_query)

    # Get the AI response based on the user's input and chat history
    with st.chat_message("AI"):
        response = get_response(user_query, st.session_state.chat_history)
        st.write(response)

    # Append the AI's response to the chat history
    st.session_state.chat_history.append(AIMessage(content=response))
