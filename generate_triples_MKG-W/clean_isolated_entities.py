import json

triples_file = "../outputs/surface_sub_dedup_outputs/MKG-W_triples_aligned.txt"
new_json = "../outputs/new_entities_relations/last_new_entities_relations.json"
output_file = "../outputs/surface_sub_dedup_outputs/MKG-W_triples_cleaned.txt"


with open(new_json, "r", encoding="utf-8") as f:
    data = json.load(f)


new_entities = set([e.strip('"') for e in data["new_entities"]])

triples = []
with open(triples_file, "r", encoding="utf-8") as f:
    for line in f:
        parts = [x.strip() for x in line.strip().split(",")]
        if len(parts) == 3:
            triples.append(tuple(parts))

from collections import Counter
cnt = Counter()

for h, r, t in triples:
    cnt[h] += 1
    cnt[t] += 1


isolated_new_entities = {e for e in new_entities if cnt[e] <= 1}

cleaned = []
for h, r, t in triples:
    if h not in isolated_new_entities and t not in isolated_new_entities:
        cleaned.append((h, r, t))

with open(output_file, "w", encoding="utf-8") as f:
    for h, r, t in cleaned:
        f.write(f"{h}, {r}, {t}\n")

