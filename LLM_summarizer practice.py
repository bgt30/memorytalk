from openai import OpenAI
from dotenv import load_dotenv
import os

# Load environment variables and API key
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Constants for the file paths and summarization parameters
FILE_PATH = "dataset/conversations2.csv"
OUTPUT_FILE_PATH = "dataset/summary_conversations4.txt"
MAX_TOKENS = 150
CHUNK_SIZE = 1000

# Function to load the latest portion of the text from a file
def load_latest_text(file_path, read_size=1000000):
    with open(file_path, 'r', encoding='utf-8') as file:
        file.seek(0, os.SEEK_END)  # Move to the end of the file
        file_size = file.tell()  # Get file size
        start_pos = max(file_size - read_size, 0)  # Calculate start position
        file.seek(start_pos, os.SEEK_SET)  # Move to start position
        text = file.read()  # Read the content from start_pos
    return text

# Function to summarize a chunk of text using the OpenAI API
def summarize_chunk(chunk, max_tokens):
    prompt = (f"""대화 파일을 챗봇 학습 데이터를 위해 모든 정보를 명확하고 정확하게 요약해 주세요. 요약 후에는 한국어 맞춤법 검사를 해주세요.
              :\n\n{chunk}""")
    response = client.completions.create(
        model="gpt-3.5-turbo-instruct",
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=0.5,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0
    )
    return response.choices[0].text.strip()

# Function to summarize the entire text by breaking it into chunks
def summarize_text(text, max_tokens=MAX_TOKENS, chunk_size=CHUNK_SIZE):
    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    summaries = [summarize_chunk(chunk, max_tokens) for chunk in chunks]
    return ' '.join(summaries)

# Function to save the summary to a file
def save_summary(summary, output_file_path):
    with open(output_file_path, 'w', encoding='utf-8') as file:
        file.write(summary)

# Main logic to load, summarize, and save the file contents
if __name__ == "__main__":
    text = load_latest_text(FILE_PATH, read_size=1000000)
    summary = summarize_text(text)
    save_summary(summary, OUTPUT_FILE_PATH)
    print("Summary saved to", OUTPUT_FILE_PATH)