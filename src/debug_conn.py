import chromadb
import os
import requests

print("--- Checking Env ---")
print(f"HTTP_PROXY: {os.environ.get('HTTP_PROXY')}")
print(f"HTTPS_PROXY: {os.environ.get('HTTPS_PROXY')}")
print(f"NO_PROXY: {os.environ.get('NO_PROXY')}")

print("\n--- Checking ChromaDB ---")
try:
    client = chromadb.HttpClient(host='2080ti', port=8001)
    print(f"Client created: {client}")
    # Try a simple call
    print("Attempting list_collections...")
    cols = client.list_collections()
    print(f"Collections: {cols}")
except Exception as e:
    print(f"ChromaDB Error: {e}")

print("\n--- Checking Ollama (Python) ---")
try:
    r = requests.get("http://2080ti:11434/api/tags", timeout=5)
    print(f"Ollama Status: {r.status_code}")
    print(f"Ollama JSON: {r.json().keys()}")
except Exception as e:
    print(f"Ollama Error: {e}")
