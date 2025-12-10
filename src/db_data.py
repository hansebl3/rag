import chromadb

# 1. DB 연결 (8001번 포트)
client = chromadb.HttpClient(host='2080ti', port=8001)
collection = client.get_collection(name="factory_manuals")

# 2. 데이터 조회 (get)
# limit=5 : 5개만 보여줘
# include=['embeddings', 'documents', 'metadatas'] : 숫자, 문자, 꼬리표 다 보여줘
data = collection.get(limit=5, include=['embeddings', 'documents', 'metadatas'])

print(f"=== 총 데이터 개수: {collection.count()}개 ===\n")

# 3. 하나씩 꺼내서 출력해보기
for i in range(len(data['ids'])):
    print(f"--- 데이터 {i+1} ---")
    print(f"🆔 ID   : {data['ids'][i]}")
    print(f"📄 문자 : {data['documents'][i]}")  # <--- 문자로 나옴 (우리가 쓸 거)
    print(f"🏷️ 메타 : {data['metadatas'][i]}")
    
    # 숫자는 너무 기니까(768개) 앞부분 5개만 출력
    vector_sample = data['embeddings'][i][:5]
    print(f"🔢 숫자 : {vector_sample} ... (총 {len(data['embeddings'][i])}개 실수)")
    print("")