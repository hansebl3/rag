import requests
import json
import chromadb
from sentence_transformers import SentenceTransformer
from datetime import datetime

# --- 설정 (환경에 맞게 IP 수정) ---
# GPU 서버 IP가 192.168.0.X라면 localhost 대신 IP 입력
OLLAMA_URL = "http://2080ti:11434/api/chat"
CHROMA_HOST = '2080ti' 
CHROMA_PORT = 8001
EMBED_MODEL_ID = 'jhgan/ko-sroberta-multitask'
LLM_MODEL = "qwen3:8b" 

class FactoryAnalyst:
    def __init__(self):
        print("🏭 공장 분석 시스템 초기화 중...")
        self.embed_model = SentenceTransformer(EMBED_MODEL_ID)
        self.db_client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
        self.collection = self.db_client.get_collection(name="factory_manuals")
        print("✅ 시스템 준비 완료.")

    def get_manual_info(self, query):
        # 질문(에러코드 등)을 벡터로 바꿔서 DB 검색
        vec = self.embed_model.encode(query).tolist()
        results = self.collection.query(query_embeddings=[vec], n_results=1)
        
        if results['documents'][0]:
            return results['documents'][0][0]
        else:
            return "관련 매뉴얼 없음."

    def generate_report(self, daily_logs):
        # 1. 로그에서 검색 키워드 뽑기 (가장 심각한 에러 기준)
        error_keyword = f"{daily_logs['critical_error']} {daily_logs['symptom']}"
        print(f"\n🔎 검색 키워드: '{error_keyword}'")

        # 2. 벡터 DB(매뉴얼) 검색
        manual_context = self.get_manual_info(error_keyword)
        print(f"📚 참고 매뉴얼: {manual_context[:50]}...")

        # 3. 프롬프트 작성 (JSON + RAG)
        system_prompt = f"""
        당신은 공장 설비 분석 AI입니다.
        
        [참고 매뉴얼]
        {manual_context}
        
        위 매뉴얼을 근거로, 아래 [일일 로그]를 분석하여 '일일 운전 리포트'를 작성하세요.
        반드시 JSON 포맷으로 출력하세요.
        """

        user_content = json.dumps(daily_logs, indent=2, ensure_ascii=False)

        # 4. Ollama에게 전송
        payload = {
            "model": LLM_MODEL,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ],
            "format": "json", # JSON 강제 출력
            "stream": False
        }

        try:
            print("🤔 AI가 분석 중입니다...")
            response = requests.post(OLLAMA_URL, json=payload).json()
            return response['message']['content']
        except Exception as e:
            return f"통신 에러: {e}"

# --- 실행 ---
if __name__ == "__main__":
    analyst = FactoryAnalyst()

    # [상황] 오늘 공장에서 발생한 로그 데이터 (가정)
    today_log = {
        "date": datetime.now().strftime("%Y-%m-%d"),
        "machine_id": "Press-01",
        "operation_hours": 8.5,
        "avg_temp": 82.1,  # 좀 높음
        "critical_error": "E-501",
        "symptom": "Overheat",
        "error_count": 3
    }

    report = analyst.generate_report(today_log)
    
    print("\n📋 [일일 분석 리포트 결과]")
    print(report)