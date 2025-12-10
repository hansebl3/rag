import streamlit as st
import os
import json
import requests
import chromadb
import pandas as pd
from sentence_transformers import SentenceTransformer

# Page Config
st.set_page_config(page_title="RAG Testing App", layout="wide")

# Custom CSS for Title (Matching News Reader)
st.markdown("""
<style>
h1 { font-size: 1.8rem !important; }
h2 { font-size: 1.5rem !important; }
</style>
""", unsafe_allow_html=True)

st.title("🛠️ LLM RAG Testing Workbench")

# System Prompt File Path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SYSTEM_PROMPT_PATH = os.path.join(SCRIPT_DIR, 'system.txt')

# Configuration for Embedding (Adjust path if needed)
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'Data', 'vectorDB')
EMBED_MODEL_ID = 'jhgan/ko-sroberta-multitask'
CHROMA_HOST = '2080ti'
CHROMA_PORT = 8001
COLLECTION_NAME = "factory_manuals"
OLLAMA_URL = "http://2080ti:11434/api/chat"
LLM_MODEL = "gpt-oss:20b"

@st.cache_resource
def get_embedding_model():
    return SentenceTransformer(EMBED_MODEL_ID)

@st.cache_resource
def get_chroma_client():
    return chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["📝 System Prompt", "💾 Vector DB Embedder", "🗃️ DB Viewer", "💬 RAG Chat"])

# ... (Previous Page Config & Title)

# ... (Tabs)

# --- Tab 2: Vector DB Embedder ---
with tab2:
    st.header("💾 Vector DB Embedder")
    
    # Check Data Directory
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR, exist_ok=True)
        st.info(f"Created data directory: {DATA_DIR}")

    # File Management UI
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # List JSON files
        files = [f for f in os.listdir(DATA_DIR) if f.endswith('.json')]
        
        # New File Input
        new_filename = st.text_input("New File Name (e.g., data_v2.json)")
        if st.button("➕ Create New File"):
            if new_filename:
                if not new_filename.endswith('.json'):
                    new_filename += '.json'
                new_path = os.path.join(DATA_DIR, new_filename)
                if os.path.exists(new_path):
                    st.error("File already exists!")
                else:
                    with open(new_path, 'w', encoding='utf-8') as f:
                        json.dump([], f, indent=4) # Init with empty list
                    st.success(f"Created {new_filename}")
                    st.rerun() # Refresh to show in list

        if not files:
            st.warning(f"⚠️ No .json files found in {DATA_DIR}")
            selected_file = None
        else:
            selected_file = st.selectbox("Select JSON File", files)

    # File Editor & Embedder
    if selected_file:
        file_path = os.path.join(DATA_DIR, selected_file)
        
        # Load Content for Editing
        if "editor_content" not in st.session_state or st.session_state.get("current_file") != selected_file:
            with open(file_path, 'r', encoding='utf-8') as f:
                st.session_state.editor_content = f.read()
            st.session_state.current_file = selected_file

        with st.expander("� Edit JSON Content", expanded=True):
            edited_content = st.text_area("JSON Data", value=st.session_state.editor_content, height=300)
            
            if st.button("💾 Save Changes"):
                try:
                    # Validate JSON
                    json.loads(edited_content)
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(edited_content)
                    st.session_state.editor_content = edited_content
                    st.success("Saved successfully!")
                except json.JSONDecodeError as e:
                    st.error(f"❌ Invalid JSON: {e}")

        st.divider()
        
        if st.button("🚀 Start Embedding"):
            # ... (Embedding Logic - existing)
            try:
                with st.status("Processing...", expanded=True) as status:
                    st.write("📖 Loading JSON data...")
                    # Reload from file to ensure saved changes are used
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    st.write(f"   -> Loaded {len(data)} items.")
                    
                    st.write(f"🧠 Loading Embedding Model ({EMBED_MODEL_ID})...")
                    model = get_embedding_model()
                    
                    st.write(f"🔌 Connecting to ChromaDB ({CHROMA_HOST}:{CHROMA_PORT})...")
                    client = get_chroma_client()
                    collection = client.get_or_create_collection(name=COLLECTION_NAME)
                    
                    st.write("🔢 Vectorizing and Upserting...")
                    docs = []
                    ids = []
                    
                    for item in data:
                        if 'text' in item and 'id' in item:
                            docs.append(str(item['text'])) # Ensure string
                            ids.append(str(item['id'])) # Ensure string ID
                    
                    if docs:
                        embeddings = model.encode(docs).tolist()
                        collection.upsert(documents=docs, embeddings=embeddings, ids=ids)
                        status.update(label="✅ Embedding Complete!", state="complete", expanded=False)
                        st.success(f"Successfully embedded {len(ids)} documents from {selected_file}")
                    else:
                        status.update(label="❌ No valid data found", state="error")
                        st.error("No items with 'text' and 'id' fields found in JSON.")
                        
            except Exception as e:
                st.error(f"❌ An error occurred: {e}")



# ... (Previous Page Config & Title)

# ... (Tabs)

# ... (Tab 1 Content)

# --- Tab 3: DB Viewer ---
with tab3:
    st.header("🗃️ Database Viewer")
    
    col_db_1, col_db_2 = st.columns([1, 1])
    
    with col_db_1:
        if st.button("🔄 Load Data (Top 5)"):
            st.session_state.db_view_mode = 'top_5'
            
    with col_db_2:
        if st.button("🔢 Load All Data"):
            st.session_state.db_view_mode = 'all'

    if "db_view_mode" in st.session_state:
        try:
            client = get_chroma_client()
            collection = client.get_collection(name=COLLECTION_NAME)
            
            count = collection.count()
            st.metric("Total Documents", count)
            
            # Determine Limit
            limit = 5 if st.session_state.db_view_mode == 'top_5' else count
            if limit == 0: limit = 1 # Avoid error if count is 0
            
            # Fetch Data
            data = collection.get(limit=limit, include=['embeddings', 'documents', 'metadatas'])
            
            # Fix for "The truth value of an array with more than one element is ambiguous"
            # We check the length explicitly and ensure it's not None.
            # Avoid `if data['ids']:` which triggers the numpy error.
            if data['ids'] is not None and len(data['ids']) > 0:
                
                # Create a DataFrame for better visualization
                df_data = []
                for i in range(len(data['ids'])):
                    # Handle Metadata
                    metadata_str = "{}"
                    if data['metadatas'] is not None and len(data['metadatas']) > i:
                         metadata_str = str(data['metadatas'][i])

                    # Handle Embeddings
                    embedding_str = "[]"
                    if data['embeddings'] is not None and len(data['embeddings']) > i:
                        # Ensure it's list-like
                        emb = data['embeddings'][i]
                        if hasattr(emb, '__len__') and len(emb) >= 3:
                            embedding_str = str(emb[:3])
                        else:
                            embedding_str = str(emb)

                    item = {
                        "ID": data['ids'][i],
                        "Document": data['documents'][i],
                        "Metadata": metadata_str,
                        "Embedding (Three dim)": embedding_str
                    }
                    df_data.append(item)
                
                df = pd.DataFrame(df_data)
                st.dataframe(df, use_container_width=True)
                
                if st.session_state.db_view_mode == 'top_5' and count > 5:
                    st.info(f"Showing top 5 of {count} documents. Click 'Load All Data' to see the rest.")
            else:
                st.warning("No data found in the collection.")
                
        except Exception as e:
            st.error(f"❌ Error connecting to DB: {e}")
            st.caption("Check if ChromaDB is running on 2080ti:8001")



# ... (Previous Page Config & Title)

# ... (Tabs)

# ... (Tab 1, 2, 3 Content)

# --- Tab 4: RAG Chat ---
with tab4:
    st.header("💬 RAG Chat")
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Sidebar Settings
    with st.sidebar:
        st.header("⚙️ Chat Settings")
        
        # Dynamic Model Fetching
        def get_ollama_models(base_url):
            try:
                # API endpoint for tags matches standard Ollama API
                # Adjust if 'http://2080ti:11434/api/tags' is the correct one
                api_tags_url = base_url.replace("/api/chat", "/api/tags")
                response = requests.get(api_tags_url, timeout=3)
                if response.status_code == 200:
                    data = response.json()
                    # Extract model names
                    return [model['name'] for model in data.get('models', [])]
            except Exception as e:
                pass
            return []

        # Try to fetch models
        available_models = get_ollama_models(OLLAMA_URL)
        
        # If fetch fails, provide a fallback list + existing default + custom option
        if not available_models:
            available_models = [LLM_MODEL, "llama3:latest", "mistral:latest", "gemma:latest"]
        
        # Ensure default model is in the list
        if LLM_MODEL not in available_models:
            available_models.insert(0, LLM_MODEL)
            
        selected_model = st.selectbox("LLM Model", available_models, index=0)
        
        # Allow custom input if needed (optional, implemented as specific choice)
        # For simplicity, we stick to selectbox for now as requested.
        
        top_k = st.slider("Top-K Retrieval", min_value=1, max_value=10, value=3)

    # Display chat messages from history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat Input
    if prompt := st.chat_input("Ask a question..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            
            try:
                # 1. Search ChromaDB
                client = get_chroma_client()
                collection = client.get_collection(name=COLLECTION_NAME)
                embed_model = get_embedding_model()
                
                query_vec = embed_model.encode(prompt).tolist()
                results = collection.query(query_embeddings=[query_vec], n_results=top_k)
                
                docs = results['documents'][0] if results['documents'] else []
                distances = results['distances'][0] if results['distances'] else []
                
                # Context Construction
                context = "\n".join([f"- {doc}" for doc in docs])
                
                # 2. Get System Prompt
                if "system_prompt_content" in st.session_state and st.session_state.system_prompt_content:
                    custom_instructions = st.session_state.system_prompt_content
                elif os.path.exists(SYSTEM_PROMPT_PATH):
                    with open(SYSTEM_PROMPT_PATH, 'r', encoding='utf-8') as f:
                        custom_instructions = f.read().strip()
                else:
                    custom_instructions = ""
                    
                system_prompt = f"""
                당신은 공장 설비 및 IP 주소 관리 전문가입니다.
                아래 [참고 정보]를 바탕으로 사용자의 질문에 답변하세요.
                
                [추가 지시사항]
                {custom_instructions}
                
                [참고 정보]
                {context}
                
                - [참고 정보]에 없는 내용은 "정보가 없습니다"라고 답하세요.
                - 답변은 간결하고 명확하게 한국어로 작성하세요.
                """
                
                # 3. Call Ollama
                payload = {
                    "model": selected_model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt}
                    ],
                    "stream": True
                }
                
                with requests.post(OLLAMA_URL, json=payload, stream=True) as r:
                    r.raise_for_status()
                    for line in r.iter_lines():
                        if line:
                            body = json.loads(line)
                            if 'message' in body:
                                content = body['message'].get('content', '')
                                full_response += content
                                message_placeholder.markdown(full_response + "▌")
                                
                message_placeholder.markdown(full_response)
                
                # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": full_response})
                
                # Optional: Show retrieved context in expander
                with st.expander("🔍 Retrieved Context"):
                    for i, doc in enumerate(docs):
                        st.markdown(f"**Doc {i+1}** (Dist: {distances[i]:.4f}):\n{doc}")

            except Exception as e:
                st.error(f"❌ Error during chat processing: {e}")
