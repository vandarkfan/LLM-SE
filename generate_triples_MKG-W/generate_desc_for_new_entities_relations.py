import json
import requests
import concurrent.futures
import os
import threading


API_URL = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
API_KEY = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

lock = threading.Lock()


def chat_with_model(model, messages, temperature=0.1):
    data = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "stream": False
    }

    try:
        response = requests.post(API_URL, headers=headers, json=data, timeout=300)
        response.raise_for_status()
        result = response.json()
        return result["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"❌ error: {e}")
        return ""



model = "Qwen3-VL-235B-A22B-Instruct"


input_path = r"../outputs/new_entities_relations/new_entities_relations.json"
output_path = r"../outputs/new_description/new_entities_relations_desc.json"

os.makedirs(os.path.dirname(output_path), exist_ok=True)


with open(input_path, "r", encoding="utf-8") as f:
    data = json.load(f)

entities = data.get("new_entities", [])
relations = data.get("new_relations", [])

print(f"📌 entity: {len(entities)} 个")
print(f"📌 realtion: {len(relations)} 个")

results = {"entities": [], "relations": []}


def process_entity(entity_name):
    prompt = f"Please provide a concise English description (under 15 words) for the entity '{entity_name}'."
    messages = [
        {"role": "system", "content": "You are a concise knowledge graph assistant."},
        {"role": "user", "content": prompt}
    ]
    desc = chat_with_model(model, messages)
    print(f"[Entity] {entity_name} → {desc}")

    with lock:
        results["entities"].append({"name": entity_name, "desc": desc})


def process_relation(relation_name):
    prompt = f"Please provide a concise English description (under 15 words) for the relation '{relation_name}'."
    messages = [
        {"role": "system", "content": "You are a concise knowledge graph assistant."},
        {"role": "user", "content": prompt}
    ]
    desc = chat_with_model(model, messages)
    print(f"[Relation] {relation_name} → {desc}")

    with lock:
        results["relations"].append({"name": relation_name, "desc": desc})



max_workers = 5


with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
    futures = [executor.submit(process_entity, e) for e in entities]
    for _ in concurrent.futures.as_completed(futures):
        pass


with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
    futures = [executor.submit(process_relation, r) for r in relations] 
    for _ in concurrent.futures.as_completed(futures):
        pass


with open(output_path, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=4, ensure_ascii=False)

print("\n🎉 Merge completed！")
print(f"📁 output file: {output_path}")
print(f"📦 Number of entities: {len(results['entities'])}")
print(f"📦 Number of relations: {len(results['relations'])}")
