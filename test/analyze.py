import requests
import json
import chromadb
from sentence_transformers import SentenceTransformer

# --- 설정 ---
OLLAMA_URL = "http://2080ti:11434/api/chat"
CHROMA_HOST = '2080ti'
CHROMA_PORT = 8001
EMBED_MODEL_ID = 'jhgan/ko-sroberta-multitask' # 저장할 때랑 같은 모델 필수!
LLM_MODEL = "qwen2.5:7b" # 사용 중인 모델명 (없으면 llama3.1 등으로 변경)

class FactoryRAG:
    def __init__(self):
        print("시스템 가동 중...")
        self.embed_model = SentenceTransformer(EMBED_MODEL_ID)
        self.db_client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
        self.collection = self.db_client.get_collection(name="factory_manuals")

    def search(self, query):
        # 질문을 벡터로 변환해서 검색
        vec = self.embed_model.encode(query).tolist()
        res = self.collection.query(query_embeddings=[vec], n_results=1)
        
        if res['documents'][0]:
            return res['documents'][0][0]
        else:
            return "관련 매뉴얼 없음."

    def analyze_log(self, log_json):
        # 1. 검색 키워드 추출 (에러코드 + 증상)
        keyword = f"{log_json.get('error_code', '')} {log_json.get('status', '')}"
        print(f"\n🔎 검색 키워드: {keyword}")

        # 2. RAG 검색
        manual_text = self.search(keyword)
        print(f"📚 참고 매뉴얼: {manual_text}")

        # 3. 프롬프트 작성
        system_prompt = f"""
        당신은 공장 설비 분석가입니다.
        아래 [매뉴얼]을 참고하여 [로그 데이터]를 분석하세요.
        반드시 JSON 포맷으로만 응답하세요.
        
        [매뉴얼]
        {manual_text}
        """

        # 4. Ollama 전송
        payload = {
            "model": LLM_MODEL,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(log_json)}
            ],
            "format": "json",
            "stream": False
        }

        try:
            resp = requests.post(OLLAMA_URL, json=payload).json()
            return resp['message']['content']
        except Exception as e:
            return f"에러 발생: {e}"

# --- 실행 ---
if __name__ == "__main__":
    app = FactoryRAG()

    # 테스트용 가짜 로그
    test_log = {
        "timestamp": "2025-12-10 10:00:00",
        "machine": "Press-01",
        "error_code": "E-501",
        "status": "Overheat",
        "current_temp": 95
    }

    print("\n🚀 분석 시작...")
    result = app.analyze_log(test_log)
    print("\n🤖 AI 분석 결과:")
    print(result)