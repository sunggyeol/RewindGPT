import os
import json
from pathlib import Path
from datetime import datetime
from langchain.docstore.document import Document
from langchain_openai import OpenAI
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from dotenv import load_dotenv
load_dotenv()

def load_history_documents(history_dir):
    documents = []
    history_path = Path(history_dir)
    
    if not history_path.exists():
        raise ValueError(f"History directory '{history_dir}' not found")
    
    for month_dir in history_path.iterdir():
        if month_dir.is_dir():
            for file_path in month_dir.glob('*.txt'):
                try:
                    with file_path.open('r', encoding='utf-8') as file:
                        content = file.read()
                        if content.strip():
                            # Store only the filename without metadata
                            documents.append(Document(
                                page_content=content,
                                metadata={'filename': file_path.name}
                            ))
                except Exception as e:
                    print(f"Error processing file {file_path}: {str(e)}")
                    continue
    
    return documents

# Environment variable for OpenAI API key
api_key = os.getenv('OPENAI_API_KEY')
if not api_key:
    raise ValueError("OPENAI_API_KEY not found in environment variables")

# Load data from history directory
documents = load_history_documents("history")
if not documents:
    raise ValueError("No documents loaded from history directory")

# Split documents
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

# Create embeddings
embeddings = OpenAIEmbeddings()

# Create and save FAISS index
db = FAISS.from_documents(docs, embeddings)
db.save_local("faiss_index")

# Initialize the LLM
llm = OpenAI(api_key=api_key)

# Memory Management
memory = ConversationBufferMemory()

# Conversation Chain
conversation = ConversationChain(llm=llm, memory=memory)

# Loading the FAISS Index
new_db = FAISS.load_local("faiss_index", embeddings)

def chatbot_response(user_input):
    relevant_docs = new_db.similarity_search(user_input)
    
    context_parts = []
    for doc in relevant_docs:
        # Get filename and extract date and topic directly from it
        filename = doc.metadata['filename']
        name_without_ext = filename.rsplit('.', 1)[0]  # Remove .txt
        date_str = '_'.join(name_without_ext.split('_')[:3])  # Get YYYY_MM_DD
        topic = '_'.join(name_without_ext.split('_')[3:])     # Get topic
        
        context_parts.append(f"[{date_str} - {topic}]\n{doc.page_content}")
    context = "\n\n".join(context_parts)
    
    prompt = f"Context from past conversations:\n{context}\n\nUser: {user_input}\nAssistant:"
    return conversation.predict(input=prompt)

# Example interaction
while True:
    user_message = input("You: ")
    if user_message.lower() == "quit":
        break
    
    response = chatbot_response(user_message)
    print(f"TalkBackGPT: {response}")

print("Goodbye!")