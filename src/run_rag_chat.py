import requests
import json
import chromadb
from sentence_transformers import SentenceTransformer
import sys
import os

# --- 설정 ---
OLLAMA_URL = "http://100.65.53.9:11434/api/chat"
CHROMA_HOST = '100.65.53.9'
CHROMA_PORT = 8001
EMBED_MODEL_ID = 'jhgan/ko-sroberta-multitask'
COLLECTION_NAME = "factory_manuals"
LLM_MODEL = "gpt-oss:20b" 

class RAGChat:
    def __init__(self):
        print("🚀 시스템 초기화 중...")
        
        # 1. 임베딩 모델 로드
        print(f"   - 임베딩 모델 로드: {EMBED_MODEL_ID}")
        try:
            self.embed_model = SentenceTransformer(EMBED_MODEL_ID)
        except Exception as e:
            print(f"❌ 임베딩 모델 로드 실패: {e}")
            sys.exit(1)

        # 2. ChromaDB 연결
        print(f"   - ChromaDB 연결: {CHROMA_HOST}:{CHROMA_PORT}")
        try:
            self.db_client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
            self.collection = self.db_client.get_collection(name=COLLECTION_NAME)
            count = self.collection.count()
            print(f"   ✅ 연결 성공! (저장된 문서: {count}개)")
        except Exception as e:
            print(f"❌ ChromaDB 연결 실패: {e}")
            print("   DB가 켜져 있는지 확인해주세요.")
            sys.exit(1)

    def search(self, query, k=3):
        """질문과 관련된 문서 Top-k 검색"""
        vec = self.embed_model.encode(query).tolist()
        res = self.collection.query(query_embeddings=[vec], n_results=k)
        
        documents = res['documents'][0]
        distances = res['distances'][0]
        
        return documents, distances

    def chat(self, user_input):
        # 1. 관련 문서 검색
        print(f"\n🔍 DB 검색 중...", end="", flush=True)
        docs, dists = self.search(user_input)
        print(" 완료.")

        if not docs:
            return "관련된 정보를 찾을 수 없습니다."

        # 검색된 내용 조합
        context = "\n".join([f"- {doc}" for doc in docs])
        
        # 2. 프롬프트 구성
        # 2. 프롬프트 구성
        # 2-1. system.txt에서 추가 지시사항 읽기
        custom_instructions = ""
        try:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            system_file_path = os.path.join(script_dir, 'system.txt')
            if os.path.exists(system_file_path):
                with open(system_file_path, 'r', encoding='utf-8') as f:
                    custom_instructions = f.read().strip()
        except Exception as e:
            print(f"⚠️ system.txt 읽기 실패 ({e})")

        # 2-2. 최종 프롬프트 조합
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

        # 3. Ollama 요청
        payload = {
            "model": LLM_MODEL,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_input}
            ],
            "stream": True # 스트리밍으로 출력
        }

        print("\n🤖 (답변 생성 중...)\n")
        
        full_response = ""
        try:
            with requests.post(OLLAMA_URL, json=payload, stream=True) as r:
                r.raise_for_status()
                for line in r.iter_lines():
                    if line:
                        try:
                            body = json.loads(line)
                            # print(f"[DEBUG] {body}") # 너무 많으면 주석 처리
                            if 'error' in body:
                                print(f"❌ Ollama 에러: {body['error']}")
                                
                            if 'message' in body:
                                content = body['message'].get('content', '')
                                print(content, end="", flush=True)
                                full_response += content
                                
                            if body.get('done', False):
                                # print("\n[DEBUG] 완료 신호 받음")
                                pass
                                
                        except json.JSONDecodeError:
                            print(f"\n❌ JSON 파싱 실패: {line}")
                            
            print("\n") # 줄바꿈
            if not full_response:
                print("⚠️ 경고: Ollama로부터 받은 응답 내용이 없습니다.")
                
            return full_response
        except Exception as e:
            print(f"\n❌ 통신 중 예외 발생: {e}")
            return f"에러: {e}"

def main():
    chat_app = RAGChat()
    
    print("\n💬 대화를 시작합니다. (종료하려면 'exit' 또는 'quit' 입력)")
    print("-" * 50)
    
    while True:
        try:
            user_input = input("\n👤 질문: ").strip()
        except KeyboardInterrupt:
            break
            
        if not user_input:
            continue
            
        if user_input.lower() in ['exit', 'quit', '종료', 'q']:
            print("👋 프로그램을 종료합니다.")
            break
            
        chat_app.chat(user_input)

if __name__ == "__main__":
    main()
