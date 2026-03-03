import requests


BACKEND_URL = "http://backend:8000/api"

class CliniQClient:
    @staticmethod
    def login(username, password):
        response = requests.post(f"{BACKEND_URL}/auth/login", data={"username": username, "password": password})
        return response

    @staticmethod
    def ask_question(token, question):
        headers = {"Authorization": f"Bearer {token}"}
        response = requests.post(f"{BACKEND_URL}/chat/ask", json={"question": question}, headers=headers)
        return response

    @staticmethod
    def get_history(token):
        headers = {"Authorization": f"Bearer {token}"}
        response = requests.get(f"{BACKEND_URL}/chat/history", headers=headers)
        return response