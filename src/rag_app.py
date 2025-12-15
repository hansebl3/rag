"""
RAG Testing Workbench
---------------------
This Streamlit app serves as a workbench for testing and verifying the RAG system.

Features:
1.  **System Prompt Editor**: Edit the instructions given to the LLM.
2.  **Vector DB Embedder**: Upload JSON files and embed them into ChromaDB.
3.  **DB Viewer**: Inspect the contents of the Vector DB collections.
4.  **RAG Chat**:
    - **Multi-Collection Search**: Query multiple knowledge bases simultaneously.
    - **ID Backtracking**: Retrieves full document context from MariaDB using `source_id` from Vector DB chunks.
    - **Context Awareness**: Displays retrieved documents and their distance scores.

Dependencies:
- streamlit, chromadb, pandas, sentence_transformers, pymysql.
"""

import streamlit as st
import os
import json
import requests
import chromadb
import pandas as pd
import pymysql # Added for ID Backtracking
from sentence_transformers import SentenceTransformer

# --- API Helper Functions ---
def call_openai_api(model, messages, api_key):
    """Generates response using OpenAI-compatible API."""
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model,
        "messages": messages,
        "stream": True
    }
    try:
        response = requests.post(url, headers=headers, json=payload, stream=True, timeout=120)
        response.raise_for_status()
        for line in response.iter_lines():
            if line:
                decoded = line.decode('utf-8')
                if decoded.startswith("data: "):
                    if decoded == "data: [DONE]":
                        break
                    try:
                        chunk = json.loads(decoded[6:])
                        if chunk['choices'] and chunk['choices'][0]['delta'].get('content'):
                            yield chunk['choices'][0]['delta']['content']
                    except:
                        pass
    except Exception as e:
        yield f"Error: {e}"

def call_gemini_api(model, messages, api_key):
    """Generates response using Google Gemini API."""
    # Convert OpenAI-style messages to Gemini 'contents'
    contents = []
    system_instruction = None
    
    for m in messages:
        role = "user" if m["role"] == "user" else "model"
        if m["role"] == "system":
            system_instruction = {"parts": [{"text": m["content"]}]}
            continue
        contents.append({"role": role, "parts": [{"text": m["content"]}]})

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:streamGenerateContent?key={api_key}"
    headers = {"Content-Type": "application/json"}
    payload = {"contents": contents}
    if system_instruction:
        payload["systemInstruction"] = system_instruction
        
    try:
        response = requests.post(url, headers=headers, json=payload, stream=True, timeout=120)
        response.raise_for_status()
        # Gemini Stream Handler
        for line in response.iter_lines():
            if line:
                try:
                    decoded = line.decode('utf-8').strip()
                    pass
                except:
                    pass
    except Exception as e:
         yield f"Error: {e}"

def call_gemini_api_non_stream(model, messages, api_key):
    # Fallback to non-streaming for reliability
    contents = []
    system_instruction = None
    for m in messages:
        role = "user" if m["role"] == "user" else "model"
        if m["role"] == "system":
            system_instruction = {"parts": [{"text": m["content"]}]}
            continue
        contents.append({"role": role, "parts": [{"text": m["content"]}]})

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    headers = {"Content-Type": "application/json"}
    payload = {"contents": contents}
    if system_instruction:
        payload["systemInstruction"] = system_instruction
        
    try:
        r = requests.post(url, headers=headers, json=payload, timeout=60)
        r.raise_for_status()
        data = r.json()
        return data['candidates'][0]['content']['parts'][0]['text']
    except Exception as e:
        return f"Error: {e}"



# MariaDB Config
DB_HOST = os.getenv("DB_HOST", "172.17.0.4") # Direct Container IP (Bridge Network)
DB_USER = os.getenv("DB_USER", "root")
DB_PASSWORD = os.getenv("DB_PASSWORD", "lavita!978")
DB_NAME = os.getenv("DB_NAME", "rag_diary_db")

def get_full_document_from_mariadb(table_name, source_id):
    """Fetches the full content from MariaDB using the source ID."""
    try:
        conn = pymysql.connect(
            host=DB_HOST, user=DB_USER, password=DB_PASSWORD, database=DB_NAME,
            charset='utf8mb4', cursorclass=pymysql.cursors.DictCursor,
            connect_timeout=5
        )
        with conn.cursor() as cursor:
            st.caption(f"🔍 Fetching valid doc from MariaDB: {source_id}") # Debug Log
            cursor.execute(f"SELECT * FROM {table_name} WHERE uuid = %s", (source_id,))
            result = cursor.fetchone()
        conn.close()
        
        if result:
            # Hybrid Schema Extraction
            content = result.get('content', '')
            
            # Handle Metadata (JSON check)
            metadata_raw = result.get('metadata')
            summary = ""
            date = result.get('log_date', '')
            subject = result.get('subject', '')
            
            if metadata_raw:
                if isinstance(metadata_raw, str):
                    try:
                        meta_dict = json.loads(metadata_raw)
                        summary = meta_dict.get('summary_ko') or meta_dict.get('summary_en') or ""
                    except:
                        pass # Raw string or parsing failed
                elif isinstance(metadata_raw, dict):
                    # PyMySQL might auto-parse JSON columns
                    summary = metadata_raw.get('summary_ko') or metadata_raw.get('summary_en') or ""
            
            # Fallback for old schema if columns exist
            if not summary:
                summary = result.get('summary_ko') or result.get('summary', '')

            full_text = f"[{table_name} / UUID:{source_id}]\nDate: {date}\nSubject: {subject}\nSummary: {summary}\n\nFull Content:\n{content}"
            return full_text
        else:
            st.error(f"❌ Document not found in MariaDB: {source_id} (Table: {table_name})")
    except Exception as e:
        st.error(f"❌ MariaDB Connection Error: {e}") 
        print(f"MariaDB Fetch Error: {e}")
    return None

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
EMBED_MODEL_ID = 'jhgan/ko-sroberta-multitask'
CHROMA_HOST = os.getenv('CHROMA_HOST', '100.65.53.9')
CHROMA_PORT = int(os.getenv('CHROMA_PORT', 8001))
COLLECTION_NAME = "tb_knowledge_base" # Unified Collection
OLLAMA_URL = "http://100.65.53.9:11434/api/chat"
LLM_MODEL = "gpt-oss:20b"

@st.cache_resource
def get_embedding_model():
    # Returns the raw SentenceTransformer model
    return SentenceTransformer(EMBED_MODEL_ID)

@st.cache_resource
def get_chroma_client():
    # Returns the low-level chromadb client
    return chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)

@st.cache_resource
def get_langchain_chroma_vectorstore():
    # Returns a LangChain Chroma vectorstore instance
    # This uses the unified COLLECTION_NAME
    return Chroma(
        client=get_chroma_client(),
        embedding_function=SentenceTransformerEmbeddings(model_name=EMBED_MODEL_ID),
        collection_name=COLLECTION_NAME
    )

# --- Global Sidebar: Scope Selection ---
with st.sidebar:
    st.header("🔍 Knowledge Base Scope")
    st.markdown("Select effective knowledge base (Category).")
    
    # Filter Categories matching Config
    scope_options = ["ALL", "Factory_Manuals", "Personal_Diaries", "Dev_Logs", "Ideas"]
    selected_scope = st.selectbox("📂 Target Category", scope_options, index=0)
    
    st.divider()

# Tabs
# Tabs
tab1, tab3, tab4 = st.tabs(["📝 System Prompt", "🗃️ DB Viewer", "💬 RAG Chat"])







# --- Tab 1: System Prompt ---
with tab1:
    st.header("📝 System Prompt")
    
    # Load existing prompt
    if "system_prompt_content" not in st.session_state:
        if os.path.exists(SYSTEM_PROMPT_PATH):
            with open(SYSTEM_PROMPT_PATH, 'r', encoding='utf-8') as f:
                st.session_state.system_prompt_content = f.read()
        else:
            st.session_state.system_prompt_content = "당신은 도움이 되는 AI 어시스턴트입니다."

    # Editor
    new_prompt = st.text_area(
        "Edit System Prompt", 
        value=st.session_state.system_prompt_content,
        height=300,
        help="This instruction will be added to the system prompt for the LLM."
    )
    
    if st.button("💾 Save System Prompt"):
        with open(SYSTEM_PROMPT_PATH, 'w', encoding='utf-8') as f:
            f.write(new_prompt)
        st.session_state.system_prompt_content = new_prompt
        st.success("System prompt saved successfully!")

# --- Tab 3: DB Viewer ---
with tab3:
    st.header(f"🗃️ Knowledge Base Viewer: {selected_scope}")
    
    # Unified Collection
    target_collection_name = COLLECTION_NAME
    
    # Filter Controls
    col_filter_1, col_filter_2 = st.columns([1, 1])
    with col_filter_1:
         # Scope is already selected in sidebar, just show it
         st.markdown(f"**Category Scope:** {selected_scope}")
    with col_filter_2:
        search_source_id = st.text_input("🔍 Filter by Source ID (UUID)", placeholder="Paste UUID here...")

    # Build Filter
    filter_conditions = []
    if selected_scope != "ALL":
        filter_conditions.append({"category": selected_scope})
    if search_source_id and search_source_id.strip():
        filter_conditions.append({"source_id": search_source_id.strip()})
        
    if len(filter_conditions) > 1:
        view_filter = {"$and": filter_conditions}
    elif len(filter_conditions) == 1:
        view_filter = filter_conditions[0]
    else:
        view_filter = None

    col_db_1, col_db_2 = st.columns([1, 1])
    
    with col_db_1:
        if st.button("🔄 Load Data (Top 5)"):
            st.session_state.db_view_mode = 'top_5'
            
    with col_db_2:
        if st.button("🔢 Load All Data"):
            st.session_state.db_view_mode = 'all'

    if "db_view_mode" in st.session_state:
        try:
            # use Unified Collection
            client = get_chroma_client()
            collection = client.get_collection(name=target_collection_name)
            
            # Count logic handling for filters involves getting IDs first or simple count() for total
            if view_filter:
                 # Chroma doesn't support count(where=...) natively without get?
                 # We will rely on the len(data['ids']) after fetch for filtered count
                 count_display_label = "Filtered"
            else:
                 count = collection.count()
                 count_display_label = "Total"
            
            # Determine Limit
            # Note: collection.count() is total size. 
            # If filtering, we might fetch less than limit anyway.
            total_count = collection.count()
            limit = 5 if st.session_state.db_view_mode == 'top_5' else total_count
            if limit == 0: limit = 1
            
            # Fetch Data with Filter
            get_kwargs = {
                "limit": limit,
                "include": ['embeddings', 'documents', 'metadatas']
            }
            if view_filter:
                get_kwargs["where"] = view_filter
                
            data = collection.get(**get_kwargs)
            
            # Correct count display for filtered view
            if view_filter and data['ids']:
                st.metric(f"Found Documents ({selected_scope} + ID Filter)", len(data['ids']))
            else:
                st.metric("Total Documents (Total)", total_count)
            
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
                         try:
                             # Sort keys for consistent display and use json dump for better readability
                             metadata_str = json.dumps(data['metadatas'][i], sort_keys=True, ensure_ascii=False)
                         except:
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
                
                if st.session_state.db_view_mode == 'top_5':
                    if view_filter:
                         if len(data['ids']) >= 5:
                            st.info(f"Showing first 5 matches. Click 'Load All Data' to see all matches.")
                    elif total_count > 5:
                        st.info(f"Showing top 5 of {total_count} documents. Click 'Load All Data' to see the rest.")
            else:
                st.warning("No data found in the collection.")
                
        except Exception as e:
            st.error(f"❌ Error connecting to DB: {e}")
            st.caption("Check if ChromaDB is running on 2080ti:8001")


    st.divider()
    st.header("🗑️ Data Delete")
    
    col_del_1, col_del_2 = st.columns([1, 1])
    
    with col_del_1:
        st.subheader("Delete by ID")
        del_id = st.text_input("Enter ID to delete")
        if st.button("Delete by ID", type="primary"):
            if not del_id:
                st.warning("Please enter an ID.")
            else:
                try:
                    client = get_chroma_client()
                    collection = client.get_collection(name=COLLECTION_NAME)
                    collection.delete(ids=[del_id])
                    st.success(f"Deleted ID: {del_id} from {COLLECTION_NAME}")
                    # st.rerun() # Refresh to update table
                except Exception as e:
                    st.error(f"Error deleting ID: {e}")

    with col_del_2:
        delete_label = f"Delete All ({selected_scope})"
        st.subheader("Delete Scope Data")
        if st.button(f"⚠️ {delete_label}", type="primary"):
            try:
                client = get_chroma_client()
                collection = client.get_collection(name=COLLECTION_NAME)
                
                # Determine what to delete based on scope
                if selected_scope == "ALL":
                    # Delete EVERYTHING
                    all_data = collection.get()
                    ids_to_delete = all_data['ids']
                    confirm_msg = f"Deleted ALL {len(ids_to_delete)} documents from {COLLECTION_NAME}."
                else:
                    # Delete only matching Category
                    scope_filter = {"category": selected_scope}
                    scope_data = collection.get(where=scope_filter)
                    ids_to_delete = scope_data['ids']
                    confirm_msg = f"Deleted {len(ids_to_delete)} documents in category '{selected_scope}'."
                
                if ids_to_delete:
                    collection.delete(ids=ids_to_delete)
                    st.success(confirm_msg)
                    # st.rerun()
                else:
                    st.info(f"No documents found for scope: {selected_scope}")
            except Exception as e:
                st.error(f"Error deleting data: {e}")





# --- Tab 4: RAG Chat ---
with tab4:
    st.header("💬 RAG Chat")
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Sidebar Settings: Model Settings
    with st.sidebar:
        st.header("⚙️ Model Settings")
        
        # Load Config
        CONFIG_FILE = os.path.join(SCRIPT_DIR, "llm_config.json")
        llm_config = {"api_keys": {}, "models": {}}
        if os.path.exists(CONFIG_FILE):
             with open(CONFIG_FILE, 'r') as f:
                 try: llm_config = json.load(f)
                 except: pass

        # Provider Selection
        available_providers = ["Ollama", "OpenAI", "Gemini", "Anthropic"]
        selected_provider = st.selectbox("LLM Provider", available_providers, index=0)

        # Persistence Logic (Local Settings)
        SETTINGS_FILE = os.path.join(SCRIPT_DIR, "rag_settings.json")
        def load_settings():
            if os.path.exists(SETTINGS_FILE):
                try:
                    with open(SETTINGS_FILE, "r") as f:
                        return json.load(f)
                except: pass
            return {}
        def save_settings(provider, model):
            try:
                with open(SETTINGS_FILE, "w") as f:
                    json.dump({"selected_provider": provider, "selected_model": model}, f)
            except Exception as e:
                print(f"Failed to save settings: {e}")
        
        settings = load_settings()
        
        # Model Selection Logic
        available_models = []
        
        if selected_provider == "Ollama":
            def get_ollama_models(base_url):
                try:
                    api_tags_url = base_url.replace("/api/chat", "/api/tags")
                    response = requests.get(api_tags_url, timeout=10)
                    if response.status_code == 200:
                        data = response.json()
                        return [model['name'] for model in data.get('models', [])]
                except Exception: pass
                return []
            
            available_models = get_ollama_models(OLLAMA_URL)
            if not available_models:
                available_models = [LLM_MODEL, "llama3:latest"]
            if LLM_MODEL not in available_models:
                available_models.insert(0, LLM_MODEL)
                
        elif selected_provider == "OpenAI":
            available_models = llm_config.get("models", {}).get("openai", ["gpt-4o", "gpt-3.5-turbo"])
        elif selected_provider == "Gemini":
            available_models = llm_config.get("models", {}).get("gemini", ["gemini-1.5-flash"])
        elif selected_provider == "Anthropic":
            available_models = llm_config.get("models", {}).get("anthropic", ["claude-3-sonnet"])
            
        # Select Model
        # Restore last selection if match
        last_provider = settings.get("selected_provider", "Ollama")
        last_model = settings.get("selected_model")
        
        # If we just switched provider, default to 0
        default_index = 0
        if selected_provider == last_provider and last_model in available_models:
             default_index = available_models.index(last_model)
             
        selected_model = st.selectbox("Model", available_models, index=default_index)

        # Auto-save
        if selected_provider != last_provider or selected_model != last_model:
            save_settings(selected_provider, selected_model)
            
        # API Key Warning
        api_key = ""
        if selected_provider != "Ollama":
            key_name = selected_provider.lower()
            api_key = llm_config.get("api_keys", {}).get(key_name, "")
            if not api_key or api_key.startswith("sk-...") or api_key == "":
                st.warning(f"⚠️ No valid API Key found for {selected_provider} in llm_config.json")
                api_key_input = st.text_input("Enter API Key (Temporary)", type="password")
                if api_key_input:
                    api_key = api_key_input
        
        top_k = st.slider("Top-K Retrieval", min_value=1, max_value=10, value=3)
        
        st.divider()
        if st.button("🔌 Test Connections"):
            with st.status("Testing Connectivity...", expanded=True):
                # Test Ollama
                try:
                    st.write(f"Testing Ollama: {OLLAMA_URL}...")
                    r = requests.get(OLLAMA_URL.replace("/api/chat", "/api/tags"), timeout=5)
                    if r.status_code == 200:
                        st.success(f"✅ Ollama Connected! Found {len(r.json().get('models', []))} models.")
                    else:
                        st.error(f"❌ Ollama Error: Status {r.status_code}")
                except Exception as e:
                    st.error(f"❌ Ollama Failed: {e}")
                
                # Test Chroma
                try:
                    st.write(f"Testing ChromaDB: {CHROMA_HOST}:{CHROMA_PORT}...")
                    test_client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
                    cnt = test_client.get_collection(name=COLLECTION_NAME).count()
                    st.success(f"✅ ChromaDB Connected! Collection '{COLLECTION_NAME}' has {cnt} docs.")
                except Exception as e:
                    st.error(f"❌ ChromaDB Failed: {e}")

        st.divider()
        if st.button("🗑️ Clear Chat History"):
            st.session_state.messages = []
            st.rerun()

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
            
            with st.spinner(f"Searching knowledge base ({selected_scope})..."):
                try:
                    # 1. Embed Query
                    model = get_embedding_model()
                    query_embedding = model.encode(prompt).tolist()
                    
                    # 2. Query Unified Vector DB (Native Chroma)
                    client = get_chroma_client()
                    collection = client.get_collection(name=COLLECTION_NAME)
                    
                    FETCH_K = top_k * 5 # Fetch more to allow for filtering
                    query_kwargs = {
                        "query_embeddings": [query_embedding],
                        "n_results": FETCH_K,
                        "include": ["metadatas", "documents", "distances"]
                    }
                    
                    if selected_scope != "ALL":
                        query_kwargs["where"] = {"category": selected_scope}
                        
                    results = collection.query(**query_kwargs)
                    
                    # --- DEBUG: View Raw Search Results ---
                    with st.expander("🕵️ Search Debugger (Raw Results)", expanded=False):
                        st.write(f"**Filter Used:** {query_kwargs.get('where', 'None (ALL)')}")
                        st.write(f"**Requested K (Unique):** {top_k}")
                        st.write(f"**Fetched K (Raw):** {FETCH_K}")
                        
                        st.write("**Raw Results Keys:**")
                        st.write(list(results.keys())) # Fixed: .keys() method
                        if results['ids']:
                            st.write(f"Found {len(results['ids'][0])} matches (before deduplication).")
                            # Show first few raw results
                            for idx, id_val in enumerate(results['ids'][0][:10]):
                                dist = results['distances'][0][idx] if results['distances'] else "N/A"
                                meta = results['metadatas'][0][idx] if results['metadatas'] else {}
                                st.code(f"[{idx}] ID: {id_val}\nDistance: {dist}\nMeta: {meta}")
                        else:
                            st.error("No matches returned from ChromaDB query.")
                    # ----------------------------------------
                    
                    # 3. Process & Deduplicate Results
                    context_parts = []
                    seen_ids = set()
                    final_sources = [] # To display in debug
                    
                    if results['ids'] and results['ids'][0]:
                        ids = results['ids'][0]
                        metadatas = results['metadatas'][0]
                        documents = results['documents'][0] # Added this line
                        distances = results['distances'][0]
                        
                        for i, source_id in enumerate(ids):
                            # Stop if we found enough unique docs
                            if len(context_parts) >= top_k:
                                break
                                
                            meta = metadatas[i]
                            # 'source_id' in metadata holds the original Record UUID (e.g., from MariaDB).
                            # The 'source_id' variable from enumerate is the Vector DB's unique key (random or suffixed).
                            # We MUST use the metadata's source_id for logical deduplication.
                            real_source_id = meta.get('source_id') or source_id
                            table_name = meta.get('table_name') or COLLECTION_NAME
                            
                            if real_source_id and real_source_id not in seen_ids:
                                # Fetch Full Content from MariaDB (Single Source of Truth)
                                full_doc = get_full_document_from_mariadb(table_name, real_source_id)
                                if full_doc:
                                    context_parts.append(full_doc)
                                    seen_ids.add(real_source_id)
                        context = "\n\n---\n\n".join(context_parts)
                        
                        # 4. LLM Generation (Native Requests)
                        # System Prompt Construction
                        custom_instructions = ""
                        if os.path.exists(SYSTEM_PROMPT_PATH):
                            with open(SYSTEM_PROMPT_PATH, "r") as f:
                                custom_instructions = f.read()

                        # Refined Prompt Strategy: Move Context to User Message
                        
                        system_instructions = f"""당신은 통합 지식 베이스(Factory, Diary, Dev, Idea) 관리자입니다.
반드시 아래 제공되는 [참고 정보(Context)]를 바탕으로 사용자의 질문에 답변하세요.
[참고 정보]에 없는 내용은 "문서에 정보가 없습니다."라고 정중히 답하세요.
답변은 한국어로 명확하고 간결하게 작성하세요.
{custom_instructions if os.path.exists(SYSTEM_PROMPT_PATH) else ""}
"""

                        final_user_message = f"""[참고 정보(Context)]
{context}

---
[사용자 질문]
{prompt}
"""
                         
                        # Debug Display
                        with st.expander("🛠️ Debug: System Prompt & Context"):
                            st.text("--- SYSTEM ---")
                            st.write(system_instructions)
                            st.text("--- USER (Context + Question) ---")
                            st.code(final_user_message)

                        messages = [
                             {"role": "system", "content": system_instructions},
                             {"role": "user", "content": final_user_message}
                        ]

                        # Generate Response based on Provider
                        response_placeholder = st.empty()
                        full_response = ""
                        
                        try:
                            if selected_provider == "Ollama":
                                # Call Ollama API Directly
                                payload = {
                                    "model": selected_model,
                                    "messages": messages,
                                    "stream": True
                                }
                                r = requests.post(OLLAMA_URL, json=payload, stream=True, timeout=120)
                                r.raise_for_status()
                                for line in r.iter_lines():
                                    if line:
                                        body = json.loads(line)
                                        if "message" in body:
                                            content = body["message"].get("content", "")
                                            full_response += content
                                            response_placeholder.markdown(full_response + "▌")
                                response_placeholder.markdown(full_response)
                                
                            elif selected_provider == "OpenAI":
                                for chunk in call_openai_api(selected_model, messages, api_key):
                                    full_response += chunk
                                    response_placeholder.markdown(full_response + "▌")
                                response_placeholder.markdown(full_response)
                                
                            elif selected_provider == "Gemini":
                                # Using Non-Stream for reliability first
                                result = call_gemini_api_non_stream(selected_model, messages, api_key)
                                full_response = result
                                response_placeholder.markdown(full_response)
                                
                            else:
                                full_response = f"Provider {selected_provider} not yet implemented."
                                response_placeholder.markdown(full_response)

                        except Exception as e:
                            st.error(f"Generation API Error: {e}")
                            full_response = f"Error generating response: {e}"

                except Exception as e:
                    st.error(f"❌ Error during RAG: {e}")
                    full_response = "에러가 발생했습니다."

        # Add ASSISTANT message
        st.session_state.messages.append({"role": "assistant", "content": full_response})
                
