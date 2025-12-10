import json
import chromadb
import os
import sys
from sentence_transformers import SentenceTransformer

# --- 설정 ---
CHROMA_HOST = '2080ti'
CHROMA_PORT = 8001
EMBED_MODEL_ID = 'jhgan/ko-sroberta-multitask'
COLLECTION_NAME = "factory_manuals" # 기존 컬렉션 이름 사용하거나 변경 가능
DATA_DIR = '/home/ross/pythonproject/rag/Data/vectorDB'

def select_json_file():
    if not os.path.exists(DATA_DIR):
        print(f"❌ 오류: 데이터 폴더를 찾을 수 없습니다: {DATA_DIR}")
        sys.exit(1)
        
    files = [f for f in os.listdir(DATA_DIR) if f.endswith('.json')]
    if not files:
        print(f"❌ 오류: '{DATA_DIR}' 폴더에 .json 파일이 없습니다.")
        sys.exit(1)
        
    print(f"\n📂 파일 목록 ({DATA_DIR}):")
    for idx, file in enumerate(files):
        print(f"  [{idx+1}] {file}")
        
    while True:
        try:
            choice = input("\n👉 처리할 파일 번호를 입력하세요: ")
            idx = int(choice) - 1
            if 0 <= idx < len(files):
                selected_file = os.path.join(DATA_DIR, files[idx])
                print(f"✅ 선택된 파일: {files[idx]}")
                return selected_file
            else:
                print("⚠️ 잘못된 번호입니다. 다시 입력해주세요.")
        except ValueError:
            print("⚠️ 숫자를 입력해주세요.")

def load_json(file_path):
    print(f"📖 JSON 데이터 로딩 중: {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"   -> 총 {len(data)}개의 데이터 로드됨.")
    return data

def init_database():
    # 0. 파일 선택
    json_file_path = select_json_file()

    # 1. JSON 데이터 로드
    data = load_json(json_file_path)
    if not data:
        print("❌ 데이터가 없습니다.")
        return

    # 2. 임베딩 모델 로딩
    print(f"1. 임베딩 모델 로딩 중... ({EMBED_MODEL_ID})")
    model = SentenceTransformer(EMBED_MODEL_ID)

    # 3. ChromaDB 연결
    print(f"2. ChromaDB({CHROMA_HOST}:{CHROMA_PORT}) 연결 중...")
    client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
    
    # 컬렉션 가져오기 (없으면 생성)
    collection = client.get_or_create_collection(name=COLLECTION_NAME)
    print(f"   컬렉션 '{COLLECTION_NAME}' 선택 완료.")

    # 4. 데이터 전처리 (리스트 변환)
    print("3. 데이터 벡터화 및 저장 준비 중...")
    
    docs = []
    ids = []
    
    # tqdm 등 진행률 표시가 없으므로 간단히 계수
    count = 0

    for item in data:
        # data.json 구조에 따름 (id, text 필수)
        # 만약 다른 구조라면 여기서 매핑 로직 수정 필요
        if 'text' not in item or 'id' not in item:
            print(f"⚠️ 건너뜀: 'text' 또는 'id' 필드가 없는 항목 - {item}")
            continue
            
        docs.append(item['text'])
        ids.append(item['id'])
        count += 1

    if not docs:
        print("❌ 저장할 유효한 데이터가 없습니다.")
        return

    # 5. 임베딩 생성
    # 데이터가 많으면 배치 처리 필요할 수 있음 (여기선 전체 처리)
    embeddings = model.encode(docs).tolist()

    # 6. DB 적재
    # upsert를 사용하면 기존 id가 있을 경우 업데이트, 없으면 추가
    collection.upsert(documents=docs, embeddings=embeddings, ids=ids)
    
    print(f"✅ 저장 완료! (총 {len(ids)}건 upsert)")

if __name__ == "__main__":
    init_database()
