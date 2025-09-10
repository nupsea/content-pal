import requests

query = "Feel good rom com movies recent"
print("User Query:", query)

url = "http://127.0.0.1:5000/recommend"

data = {"query": query}
response = requests.post(url, json=data)

print("Response:", response.json())

