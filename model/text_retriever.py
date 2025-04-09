import json, jsonlines
import requests
from tqdm import tqdm
import os

class TextRetriver():
    def __init__(self):
        self.webs = {
            "ES": "http://192.168.1.37:9875",
            "DPR": "http://192.168.1.37:9874",
            "entity_link": "http://192.168.1.37:9876",
        }
        self.question2id = {}
        with jsonlines.open("/data/ljx/qd_with_mcts/data/question2id.jsonl", "r") as f:
            for id, question in f:
                self.question2id[question] = id
        self.f = jsonlines.open("/data/ljx/qd_with_mcts/data/question2id.jsonl", "a")
    
    def retrieve(self, question, program, source="ES", k=100):
        if question in self.question2id:
            id = self.question2id[question]
            path = f"../data/{source}_texts/{id}.json"
            if os.path.exists(path):
                texts = json.load(open(path))
                return texts
            
        if question not in self.question2id:
            id = len(self.question2id)
            self.question2id[question] = id
            self.f.write((id, question))
        if source in {"ES", "DPR"}:
            data = {
                "query": question, 
                "k": k,
            }
            r = requests.get(self.webs[source], json=data)
            contexts = [{"text": text} for text in r.json()] if source == "ES" else r.json()
            
        path = path = f"../data/{source}_texts/{id}.json"
        json.dumps(contexts, open(path, "w"), indent=4, ensure_ascii=False)
        return contexts