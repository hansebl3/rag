import chromadb
import os
import sys

# Test connection to Local ChromaDB
host = '127.0.0.1' 
port = 8001

print(f"Testing connection to ChromaDB at {host}:{port}...")

try:
    client = chromadb.HttpClient(host=host, port=port)
    print("✅ Client created successfully.")
    
    print("Attempting heartbeat...")
    hb = client.heartbeat()
    print(f"✅ Heartbeat: {hb}")
    
    print("Attempting list_collections...")
    cols = client.list_collections()
    print(f"✅ Collections: {cols}")
    
except Exception as e:
    print(f"❌ Verification Failed: {e}")
    sys.exit(1)

print("\nMigration Verification Passed!")
