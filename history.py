import json
import datetime
import uuid
import os

class ChatHistory:
    def __init__(self, user_id, base_filename="chat_history"):
        self.user_id = user_id
        self.filename = f"{base_filename}_{user_id}.json"
        self.data = self.load_data()

    def load_data(self):
        try:
            with open(self.filename, "r", encoding="utf-8") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {"user_id": self.user_id, "history": []}

    def save_data(self):
        with open(self.filename, "w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=4, ensure_ascii=False)

    def add_message(self, message_type, content):
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        message_id = str(uuid.uuid4())
        self.data["history"].append({
            "id": message_id,
            "timestamp": timestamp,
            "type": message_type,
            "content": content
        })
        self.save_data()

    def get_history(self):
        return self.data["history"]

    def clear_history(self):
        self.data["history"] = []
        self.save_data()

    def delete_message(self, message_id):
        self.data["history"] = [msg for msg in self.data["history"] if msg["id"] != message_id]
        self.save_data()