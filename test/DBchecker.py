import chromadb

try:
    # 1. DB 연결
    client = chromadb.HttpClient(host='2080ti', port=8001)
    
    # 2. 컬렉션 가져오기
    # (주의: 저장할 때 썼던 이름 'factory_manuals'와 똑같아야 함)
    collection = client.get_collection("factory_manuals")
    
    # 3. 개수 세기
    count = collection.count()
    
    print(f"--------")
    print(f"📊 현재 저장된 데이터 개수: {count}개")
    print(f"--------")
    
    if count > 0:
        # 데이터가 있으면 첫 번째 거 하나만 맛보기로 출력
        peek = collection.peek(limit=1)
        print(f"👀 맛보기 데이터: {peek['documents'][0]}")
    else:
        print("⚠️ 데이터가 0개입니다. 저장이 안 됐거나, 컬렉션 이름이 다릅니다.")

except Exception as e:
    print(f"❌ 에러 발생: {e}")