import json
import os

DATA_DIR = '/home/ross/pythonproject/rag/Data/vectorDB'
TARGET_FILES = ['ipList.json', 'mappingTable.json']

def flatten_json(y):
    out = {}

    def flatten(x, name=''):
        if type(x) is dict:
            for a in x:
                flatten(x[a], name + a + '.')
        elif type(x) is list:
            for i, a in enumerate(x):
                flatten(a, name + str(i) + '.')
        else:
            out[name[:-1]] = x

    flatten(y)
    return out

def convert_file(filename):
    file_path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(file_path):
        print(f"⚠️ 파일 없음: {filename}")
        return

    print(f"Processing {filename}...")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            print(f"❌ JSON 파싱 에러: {filename}")
            return

    # 이미 변환된 형식인지 확인 (list이고 요소가 id, text를 가짐)
    if isinstance(data, list) and len(data) > 0 and 'id' in data[0] and 'text' in data[0]:
        print(f"ℹ️ 이미 변환된 파일인 것 같습니다: {filename}")
        return

    # 중첩 딕셔너리 평탄화
    flat_data = flatten_json(data)
    
    # 변환된 리스트 생성
    converted_list = []
    for key, value in flat_data.items():
        # key: "workcells.workcell_1.group_1.Master PC"
        # value: "192.168.1.16"
        
        # 텍스트는 key의 마지막 부분과 value를 조합하여 의미 있게 만듦
        # 예: "Master PC: 192.168.1.16"
        last_key = key.split('.')[-1]
        text_content = f"{last_key}: {value}"
        
        entry = {
            "id": key,
            "text": text_content,
            # 원본 전체 경로를 메타데이터로 남기고 싶다면 아래 주석 해제
            # "metadata": {"path": key, "original_value": value} 
        }
        converted_list.append(entry)

    # 원본 파일 덮어쓰기 (백업은 선택사항, 여기선 바로 덮어씀)
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(converted_list, f, indent=2, ensure_ascii=False)
    
    print(f"✅ 변환 완료: {filename} (총 {len(converted_list)} 항목)")

if __name__ == "__main__":
    for target in TARGET_FILES:
        convert_file(target)
