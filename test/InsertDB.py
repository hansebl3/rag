import chromadb
from sentence_transformers import SentenceTransformer

# --- 설정 ---
CHROMA_HOST = '2080ti'
CHROMA_PORT = 8001
EMBED_MODEL_ID = 'jhgan/ko-sroberta-multitask' # 한국어 최적화 모델

def init_database():
    print(f"1. 임베딩 모델 로딩 중... ({EMBED_MODEL_ID})")
    # RTX 2060을 임베딩 전용으로 쓰려면 device='cuda:1'로 변경하세요.
    # 지금은 테스트니 기본(CPU/GPU자동)으로 둡니다.
    model = SentenceTransformer(EMBED_MODEL_ID)

    print("2. ChromaDB(8001) 연결 중...")
    client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
    
    # 기존 컬렉션 있으면 삭제 후 새로 생성 (테스트용 초기화)
    try:
        client.delete_collection("factory_manuals")
        print("   기존 데이터 삭제 완료.")
    except:
        pass
    
    collection = client.create_collection(name="factory_manuals")

    # --- 샘플 데이터 (매뉴얼) ---
    manuals = [
        {
            "id": "ERR_501",
            "text": "E-501 에러는 '히터 과열'입니다. 설정 온도가 80도를 넘는지 확인하고, 냉각 팬(Cooling Fan)의 전원을 껐다 켜보세요."
        },
        {
            "id": "ERR_002",
            "text": "진동 수치가 0.5 이상이면 '베어링 마모'가 의심됩니다. 즉시 가동을 멈추고 예비 베어링으로 교체하십시오."
        }
    ]

    print("3. 데이터 벡터화 및 저장 중...")
    
    # 리스트로 변환
    docs = [m["text"] for m in manuals]
    ids = [m["id"] for m in manuals]
    embeddings = model.encode(docs).tolist()

    # DB 적재
    collection.add(documents=docs, embeddings=embeddings, ids=ids)
    
    print(f"✅ 저장 완료! (총 {len(ids)}건)")

if __name__ == "__main__":
    init_database()