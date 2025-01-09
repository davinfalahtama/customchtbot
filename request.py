import requests
import json

url = "http://127.0.0.1:8000/chat"
data = {
    "question": "Apa saja layanan yang ditawarkan Indonesia AI?",
    "chat_history": []
}
headers = {'Content-type': 'application/json'}

response = requests.post(url, data=json.dumps(data), headers=headers)

if response.status_code == 200:
    print(response.json())
else:
    print(f"Error: {response.status_code} - {response.text}")