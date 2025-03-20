from dotenv import load_dotenv

# Load the model and tokenizer
#model_name = "ft:gpt-3.5-turbo-0125:personal::9hWJQiyN" 태두어머니
#model_name = "ft:gpt-3.5-turbo-0125:personal::9rikV58V" #김재훈
#model_name = "ft:gpt-3.5-turbo-0125:personal::9ujnzCrY" # test 1
#model_name = "ft:gpt-3.5-turbo-0125:personal::9ujzN4AP" # test 2
#model_name = "ft:gpt-3.5-turbo-0125:personal::9zzGCZnN"  # test 3
#model_name = "ft:gpt-3.5-turbo-0125:personal::A1VYSm78"  # test 4
model_name  = "ft:gpt-3.5-turbo-0125:personal::A44xihV0"

import time
import random

# .env 파일에서 환경 변수 로드
load_dotenv()

# OpenAI API 키 설정

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.text_splitter import RecursiveCharacterTextSplitter
# 수정된 임포트 경로
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser



# wrap the model in a langchain llm class
llm = ChatOpenAI(model_name=model_name, temperature=0.5, max_tokens=100)

# Output parser
output_parser = StrOutputParser()


# Read the conversation data from txt file
conversation_data = open('dataset/summary_conversations.txt', 'r', encoding='utf-8').read()
conversation_history = ''
for line in conversation_data.split('\n'):
    conversation_history += line + '\n'



text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 100
)

all_splits = text_splitter.create_documents([conversation_history])

print(len(all_splits))

vector_store = Chroma.from_documents(
    documents = all_splits,
    embedding = OpenAIEmbeddings()
)

retriever = vector_store.as_retriever()

def get_current_time():
    return time.strftime("%Y-%m-%d %H:%M:%S %A", time.localtime())

# Wrap your llm_chain response with a function that selects from diverse options
def diversify_response(message, context, history_langchain_format):
    possible_responses = []
    for _ in range(3):  # Generate multiple responses
        response = llm_chain.invoke({"message": message, "time": get_current_time(), "history": history_langchain_format, "in_context": is_in(message, context), "context": context})
        if response not in possible_responses:
            possible_responses.append(response)
    
    return random.choice(possible_responses)

# Function to check if the response is repetitive
def is_repetitive(response, history):
    for _, ai in history:
        if ai == response:
            return True
    return False


# RAG or Guess?
def is_in(question, context):
    is_in_conetxt = """As a helpful assistant, 
please use your best judgment to determine if the answer to the question is within the given context. 
If the answer is present in the context, please respond with "yes". 
If not, please respond with "no". 
Only provide "yes" or "no" and avoid including any additional information. 
Please do your best. Here is the question and the context:
---
CONTEXT: {context}
---
QUESTION: {question}
---
OUTPUT (yes or no):"""

    is_in_prompt = ChatPromptTemplate.from_template(is_in_conetxt)
    chain = is_in_prompt | ChatOpenAI() | StrOutputParser()

    response = chain.invoke({"context": context, "question": question})
    print(response)
    return response.lower().startswith("yes")

# create a prompt template
template =[
    ("system","""
    You are a chatbot who is imitating a person. You should appropriately generate responses as if you were that person.
    Please provide most correct answer from the following context. 
    Think step by step and look the chat history and context carefully to provide the most correct answer in Korean.

    Sometimes you should ask a question to user for natural conversation.
    Here is current time and date information: {time}
    When you are asked questions about time or date, based on the current time and date information, think step by step and answer the question.
    You should consider only the time and date information provided in the prompt, not in context.
    If you are asked a question about the user's personal information, you should answer based on the context provided in the prompt.
    
    ---
    context: {context}
   
    
    
    
    You should generate a response that fit the following format.
    example1
     ------------------------------------
     user: 엄마 지금 뭐해?
        assistant: 요리하고 있어 너는?
        ------------------------------------
    example2
        ------------------------------------
     user: 형 올 때 물 좀 사와
        assistant: 몇리터로?
        ------------------------------------
    example3
     ------------------------------------
     user: 오늘 저녁 뭐 먹을까?
        assistant: 뭐 먹고 싶어?
        ------------------------------------
     
    Avoid repeating the same answer considering the following chat history
"""),
    MessagesPlaceholder(variable_name="history"),
    ("user","{message}"),
]

prompt_template = ChatPromptTemplate.from_messages(template)


llm_chain = prompt_template | llm | output_parser


import gradio as gr

def chat(message, history):
    history_langchain_format = []
    print(history)
    for human, ai in history:
        history_langchain_format.append({"role": "user", "content": human})
        history_langchain_format.append({"role": "assistant", "content": ai})

    context = retriever.invoke(message)
    print(get_current_time())
    print(message)
    in_context = is_in(message, context)
    print(in_context)
    response = llm_chain.invoke({"message": message, "time": get_current_time(), "history": history_langchain_format, "in_context": in_context, "context": ""})
    # Generate response based on the context if the context is relevant to the question
    if in_context:
        response = llm_chain.invoke({"message": message, "time": get_current_time(), "history": history_langchain_format, "in_context": in_context, "context": context})
        print(context)
    print(response)


    # Check if response is repetitive
    attempts = 0
    max_attempts = 3
    while is_repetitive(response, history) and attempts < max_attempts:
        response = llm_chain.invoke({"message": message, "time": get_current_time(), "history": history_langchain_format, "in_context": in_context, "context": context})
        attempts += 1

    return response
    
with gr.Blocks() as demo:
    chatbot = gr.ChatInterface(
        chat,
        title = "MemoryTalk",
        description = "잊지 못한 소중한 그 분과 대화해 보세요"
    )
    chatbot.chatbot.height = 300


if __name__ == "__main__":
    demo.launch(share = True)


