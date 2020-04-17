import requests
import json
r  = requests.post("http://localhost:12847/intent", data=json.dumps({"utterance":"hi"}),headers={"Content-Type": "application/json"})
print(r)
final = r.json()["intent"]
confidence = r.json()["confidence"]