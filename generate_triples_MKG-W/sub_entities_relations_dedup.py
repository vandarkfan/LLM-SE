import json
import torch
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm

input_path = r"../outputs/new_description/new_entities_relations_desc.json"

output_path = r"../outputs/dedup_map/new_entities_relations_dedup_map.json"

similarity_threshold = 0.8
model_name = "sentence-transformers/all-mpnet-base-v2"



with open(input_path, "r", encoding="utf-8") as f:
    data = json.load(f)

entity_items = data.get("entities", [])
relation_items = data.get("relations", [])

print(f"📌 entity: {len(entity_items)}")
print(f"📌 relation: {len(relation_items)}")

def dedup_items(items, threshold=0.8):
    """items: [{'name': str, 'desc': str}, ...]"""
    names = []
    texts = []

    for item in items:
        n = item["name"].strip()
        d = item.get("desc", "").strip()
        t = f"{n}. {d}" if d else n
        names.append(n)
        texts.append(t)

    num_original = len(names)


    model = SentenceTransformer(model_name)
    embeddings = model.encode(
        texts,
        convert_to_tensor=True,
        show_progress_bar=True,
        normalize_embeddings=True
    )


    sim_matrix = util.cos_sim(embeddings, embeddings)
    N = len(names)

    mapping = {}
    visited = set()

    print("🔍 deduplication...")
    for i in tqdm(range(N)):
        if names[i] in visited:
            continue

        similar_indices = torch.where(sim_matrix[i] > threshold)[0].tolist()
        candidate_names = [names[j] for j in similar_indices]

        main_name = max(candidate_names, key=lambda x: len(x.split()))

        for name in candidate_names:
            mapping[name] = main_name
            visited.add(name)

    # 输出统计
    unique = set(mapping.values())
    print("===================================")
    print(f"🧩 Original quantity: {num_original}")
    print(f"✅ Quantity after deduplication: {len(unique)}")
    print(f"🗑️  deduplication count: {num_original - len(unique)}")
    print(f"📉 ratio: {(num_original - len(unique)) / num_original:.2%}")
    print("===================================")

    return mapping



entity_map = dedup_items(entity_items, similarity_threshold)

relation_map = dedup_items(relation_items, similarity_threshold)

output = {
    "entity_map": entity_map,
    "relation_map": relation_map
}

with open(output_path, "w", encoding="utf-8") as f:
    json.dump(output, f, indent=4, ensure_ascii=False)

print(f"\n💾 save：{output_path}")
