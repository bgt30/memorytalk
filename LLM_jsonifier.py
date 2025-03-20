import re
import csv
from datetime import datetime, timedelta
import json

import pandas as pd
import os
import zipfile


# Check if the messages are consecutive
def is_consecutive_messages(time1, time2):
    # Check if the time difference between two messages is less than 1 hour
    # ex) 2021-10-12 00:00:00, 2021-10-12 00:30:00 -> True
    # ex) 2021-10-12 00:00, 2021-10-12 01:30 -> False
    # time cattribute may or may not have seconds
    has_seconds = len(time1) == 19
    time_format = '%Y-%m-%d %H:%M:%S' if has_seconds else '%Y-%m-%d %H:%M'
    prev_time = datetime.strptime(time1, time_format)
    curr_time = datetime.strptime(time2, time_format)
    return (curr_time - prev_time) <= timedelta(minutes=60)

def parse_kakao_talk(text):
    # Define regex patterns for extracting date, user, and message
    date_patterns = [
        r'[-]+ \d{4}년 \d{1,2}월 \d{1,2}일 [가-힣]+ [-]+',
        r'\d{4}\. \d{1,2}\. \d{1,2}\.',
        r'\d{4}년 \d{1,2}월 \d{1,2}일'
    ]
    message_patterns = [
        r'\[(.*?)\] \[(오전|오후) (\d{1,2}):(\d{2})\] (.*)', # Windows pattern
        r'(\d{4}\. \d{1,2}\. \d{1,2}\. (오전|오후) (\d{1,2}):(\d{2})), (.*?)(?:님이 .+?을 초대했습니다|:)(.*)', # iOS pattern
        r'(\d{4}년 \d{1,2}월 \d{1,2}일 (오전|오후) (\d{1,2}):(\d{2})), (.*?)(?:님이 .+?을 초대했습니다|:)(.*)' # Android pattern
    ]

    lines = text.splitlines()
    data = []

    current_date = None

    for line in lines:
        # Check if the line is a date line
        for date_pattern in date_patterns:
            if re.match(date_pattern, line):
                current_date_match = re.findall(r'\d{4}[년\.] \d{1,2}[월\.] \d{1,2}[일\.]', line)
                if current_date_match:
                    current_date = current_date_match[0]
                    current_date = current_date.replace('년', '-').replace('월', '-').replace('일', '').replace('.', '-').strip()
                    
                    current_date = re.sub(r'\s+', '', current_date) # Remove extra spaces
                    current_date.rstrip('-')
                    
                    if current_date[-1] == '-':
                        current_date = current_date[:-1]
                    
                    
                break
        
        # Check if the line is a message line
        for message_pattern in message_patterns:
            match = re.match(message_pattern, line)
            if match:
                groups = match.groups()
                if len(groups) == 5:
                    user, period, hour, minute, message = groups
                else:
                    period, hour, minute, user, message = groups[1], groups[2], groups[3], groups[4], groups[5]

                hour = int(hour)
                if period == '오후' and hour != 12:
                    hour += 12
                elif period == '오전' and hour == 12:
                    hour = 0
                time = f'{hour:02}:{minute}'
                datetime_str = f'{current_date} {time}'
                try:
                    datetime_obj = datetime.strptime(datetime_str, '%Y-%m-%d %H:%M')
                    formatted_date = datetime_obj.strftime('%Y-%m-%d %H:%M')
                    data.append([formatted_date, user.strip(), message.strip()])
                except ValueError as e:
                    print(f"Error parsing date: {datetime_str} with error: {e}")
                break
    
    return data

def save_to_csv(data, filename):
    with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['Date', 'User', 'Message'])
        csvwriter.writerows(data)

def convert_kakao_to_csv(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as file:
        text = file.read()

    data = parse_kakao_talk(text)
    save_to_csv(data, output_file)


# Example usage
# convert_kakao_to_csv('dataset/ForTest_3.txt', 'csv_test_3.csv')

# Convert the csv file to jsonl               
def convert_to_jsonl(input_file, output_file, user, assistant = None):
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    user = user
    assistant = assistant
    conversations = []
    prev_time = None
   # conversation format ex) {"messages": [{"role": "system", "content": "Marv is a factual chatbot that is also sarcastic."}, {"role": "user", "content": "What's the capital of France?"}, {"role": "assistant", "content": "Paris, as if everyone doesn't know that already."}]}
    conversation = {"messages": [{"role": "system", "content": "You are simulating a conversation between two people. It could be friends, family, etc. You should appropriately reason who they are. Resopond apporpriately as if you became one of them."}]}
    for line in lines:
        if line.startswith('Date'):
            continue
        line = line.strip().split(',')
        
        if len(line) == 1:
            continue
        
        # if line does not starts with date, it is a continuation of the previous message
        if not line[0].startswith('20'):
            continue

        print(line[0])
    

        if line[1].startswith('"'):
            line[1] = line[1][1:]
        if line[2].startswith('"'):
            line[2] = line[2][1:]
        if line[1].endswith('"'):
            line[1] = line[1][:-1]
        if line[2].endswith('"'):
            line[2] = line[2][:-1]

        # Skip if the message is a photo, emoticon, video, voice message or link or too short
        
        if line[2] in ['사진', '이모티콘', '동영상', '음성메시지','Photo','Emoticon','Video','삭제된 메시지입니다.'] or line[2].startswith("https") or len(line[2])<3:
            continue
        # Consecutive messages from same user are considered as one conversation
        # example: first message from user, second message from user, first message from assistant -> 2 conversations
        # one json line should be in the form :{"messages": [{"role": "system", "content": "Marv is a factual chatbot that is also sarcastic."}, {"role": "user", "content": "What's the capital of France?"}, {"role": "assistant", "content": "Paris, as if everyone doesn't know that already."},{"time": "2021-10-12 11:00 AM"}]}
        if prev_time is None or is_consecutive_messages(prev_time,line[0]):
            if line[1] == user or line[1] != assistant:
                if len(conversation["messages"]) == 1:
                    conversation["messages"].append({"role": "user", "content": line[2]})
                    conversation["messages"].append({"time": line[0]})
                elif len(conversation["messages"]) > 1:
                    if conversation["messages"][-2]["role"] == "user":
                        conversation["messages"][-2]["content"] += '\n' + line[2]
                    elif conversation["messages"][-2]["role"] == "assistant":
                        conversation["messages"].append({"role": "user", "content": line[2]})
                        conversation["messages"].append({"time": line[0]})
            
            #elif line[1] == assistant:
            else:
                # if there is no previous message from user, just continue
                if len(conversation["messages"]) == 1:
                    continue
                if len(conversation["messages"]) == 1:
                    conversation["messages"].append({"role": "assistant", "content": line[2]})
                    conversation["messages"].append({"time": line[0]})
                elif len(conversation["messages"]) > 1:
                    if conversation["messages"][-2]["role"] == "assistant":
                        conversation["messages"][-2]["content"] += '\n' + line[2]
                    elif conversation["messages"][-2]["role"] == "user":
                        conversation["messages"].append({"role": "assistant", "content": line[2]})
                        conversation["messages"].append({"time": line[0]})
                    
        
        prev_time = line[0]

        if len(conversation["messages"]) == 5:
            conversations.append(conversation)  
            conversation = {"messages": [{"role": "system", "content": "You are simulating a conversation between two people. It could be friends, family, etc. You should appropriately reason who they are. Resopond apporpriately as if you became one of them."}]}

    # Remove all the time attributes
    for conversation in conversations:
        conversation["messages"] = [message for message in conversation["messages"] if message and "time" not in message]
                

    for conversation in conversations:
        #if the conversation has only one message, remove the conversation
        if len(conversation["messages"]) == 1:
            conversations.remove(conversation)

    with open(output_file, 'w', encoding='utf-8') as f:
        count = 0     
        for conversation in conversations:
            # Up to 60 conversations
            count += 1
            if count > 60:
                break
            f.write(json.dumps(conversation, ensure_ascii=False) + '\n')



if __name__ == "__main__":

    txt_file = 'dataset/conversationruna.txt'
    csv_file = 'dataset/csv_conversationruna.csv'
    json_file = 'dataset/training_dataruna.jsonl'
    user = "루나"
    assistant = "혜윤언니"
    convert_kakao_to_csv(txt_file, csv_file)
    convert_to_jsonl(csv_file, json_file, user, assistant)



