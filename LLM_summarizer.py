from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def load_latest_text(file_path, read_size=400000000):
    with open(file_path, 'r', encoding='utf-8') as file:
        file.seek(0, os.SEEK_END)  # 파일의 끝으로 이동
        file_size = file.tell()  # 파일 크기 확인
        start_pos = max(file_size - read_size, 0)  # 읽기 시작할 위치 계산
        file.seek(start_pos, os.SEEK_SET)  # 읽기 시작할 위치로 이동
        text = file.read()
    return text

def summarize_text(text, max_tokens=150, chunk_size=1000):
    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    summaries = []

    for chunk in chunks:
        response = client.completions.create(
            model="gpt-3.5-turbo-instruct",
            prompt=f"""대화 파일을 챗봇 학습 데이터를 위해 모든 정보를 명확하고 정확하게 요약해 주세요. 요약 후에는 한국어 맞춤법 검사를 해주세요.
                    :\n\n{chunk}""",
            max_tokens=max_tokens,
            temperature=0.5,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0
        )
        summary = response.choices[0].text.strip()
        summaries.append(summary)

    final_summary = ' '.join(summaries)
    return final_summary

def save_summary(summary, output_file_path):
    with open(output_file_path, 'w', encoding='utf-8') as file:
        file.write(summary)

file_path = "dataset/conversations2.csv"
output_file_path = "dataset/summary_conversations4_1.txt"

# 최신 부분만 로드
text = load_latest_text(file_path, read_size=1000000)
summary = summarize_text(text)
save_summary(summary, output_file_path)

print("Summary saved to", output_file_path)