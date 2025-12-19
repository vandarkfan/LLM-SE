import os
import base64
from openai import OpenAI
import re
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import ast
import json

import requests
import json

API_URL = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
API_KEY = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

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
        content = result["choices"][0]["message"]["content"]
        return content.strip()
    return ""

model1 = "Qwen3-VL-235B-A22B-Instruct"
# model2 = "Qwen3-Max"
# model1 = "qwen3-vl-plus"
# model2 = "qwen3-max"
entity_map_file = "../train_map/train_entity_map.txt"
relation_map_file = "../train_map/train_relation_map.txt"
description_file = "../MKG-W_ent2description.txt"
image_root = "D:/IJCAI/datasets/DW_open"
wikidata_embeddings_file = "../wikidata_all_properties_20220706+embeddings.csv"
output_file = "../outputs/ori_triples/MKG-W_triples1.txt"
faiss_index_file = "../wikidata_relations.index"
wikidata_pickle_file = "../wikidata_df.pkl"

wikidata_df = pd.read_csv(wikidata_embeddings_file)


def parse_embedding(x):
    if isinstance(x, str):
        # 去除所有 np.float32( ) 包裹，只保留数字内容
        x = re.sub(r'np\.float32\((.*?)\)', r'\1', x)
        try:
            return np.array(ast.literal_eval(x), dtype=np.float32)
        except Exception as e:
            print(f"error {x[:100]}... => {e}")
            return np.nan
    return np.nan
if os.path.exists(faiss_index_file) and os.path.exists(wikidata_pickle_file):
    index = faiss.read_index(faiss_index_file)
    wikidata_df = pd.read_pickle(wikidata_pickle_file)
else:
    print("📖 load Wikidata...")
    wikidata_df = pd.read_csv(wikidata_embeddings_file)
    wikidata_df["embedding"] = wikidata_df["embedding"].apply(parse_embedding)
    
    print("🔧 FAISS...")
    embeddings_matrix = np.vstack(wikidata_df['embedding'].values).astype('float32')
    dimension = embeddings_matrix.shape[1]
    index = faiss.IndexFlatIP(dimension)
    faiss.normalize_L2(embeddings_matrix)
    index.add(embeddings_matrix)
    
    print("💾 save...")
    faiss.write_index(index, faiss_index_file)
    wikidata_df.to_pickle(wikidata_pickle_file)
    print(f"✅ FAISS save")

encoder_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')


def normalize_entity_name(name: str):
    if not name:
        return ""

    # 去掉首尾空格
    name = name.strip()

    # 1. 把下划线替换为空格
    name = name.replace("_", " ")

    # 2. 去掉括号及其内部内容  ( ... )  [ ... ]  { ... }
    name = re.sub(r"\(.*?\)|\[.*?\]|\{.*?\}", "", name)

    # 3. 去掉多余空格
    name = re.sub(r"\s+", " ", name).strip()

    return name
def extract_entity_from_url(url):
    return url.split('/')[-1]

def load_entity_map(map_file):
    entities = []
    with open(map_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(', ', 1)
            if len(parts) == 2:
                entities.append((parts[0], parts[1]))
    return entities

def load_relation_map(map_file):
    relations = []
    with open(map_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(', ', 1)
            if len(parts) == 2:
                relations.append((parts[0], parts[1]))
    return relations

entity_list = load_entity_map(entity_map_file)
entity_set = set([name for _, name in entity_list])
relation_list = load_relation_map(relation_map_file)
relation_set = set([name for _, name in relation_list])


def load_descriptions(desc_file):
    descriptions = {}
    with open(desc_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split('\t', 1)
            if len(parts) >= 2:
                url = parts[0]
                desc = parts[1].strip().strip('"').replace('@en .', '').strip()

                entity_from_url = extract_entity_from_url(url)

                normalized = normalize_entity_name(entity_from_url)

                descriptions[normalized] = (url, desc)
    return descriptions

def encode_image_to_base64(img_path):
    with open(img_path, "rb") as f:
        ext = os.path.splitext(img_path)[1].lower()
        mime = "jpeg" if ext in [".jpg", ".jpeg"] else "png"
        return f"data:image/{mime};base64," + base64.b64encode(f.read()).decode("utf-8")

def find_images_for_entity(entity_name, image_root):
    normalized = normalize_entity_name(entity_name)

    possible_names = {
        entity_name,
        entity_name.replace(" ", "_"),
        normalized,
        normalized.replace(" ", "_"),
    }

    for name in possible_names:
        folder_path = os.path.join(image_root, name)
        if os.path.exists(folder_path):
            images = [
                os.path.join(folder_path, img)
                for img in os.listdir(folder_path)
                if img.lower().endswith((".jpg", ".jpeg", ".png"))
            ]
            if images:
                return images  # 返回所有图片

    return []  # 无图片返回空列表

def parse_triples(raw_output):
    lines = raw_output.strip().split('\n')
    triples = []
    
    for line in lines:
        line = line.strip()
        if not line or line.startswith('#') or line.startswith('###'):
            continue
        
        line = re.sub(r'^\d+[\.\)]\s*', '', line)
        line = re.sub(r'^\[|\]$', '', line)
        
        parts = [p.strip() for p in line.split(',')]
        
        if len(parts) == 4:
            triples.append((parts[0], parts[1], parts[2]))
    
    return triples
def parse_triples_with_source(raw_output):
    lines = raw_output.strip().split('\n')
    triples = []
    
    for line in lines:
        line = line.strip()
        if not line or line.startswith('#') or line.startswith('###'):
            continue
        
        line = re.sub(r'^\d+[\.\)]\s*', '', line)
        line = re.sub(r'^\[|\]$', '', line)
        
        parts = [p.strip() for p in line.split(',')]
        
        if len(parts) == 4:
            triples.append((parts[0], parts[1], parts[2], parts[3]))
    
    return triples
def find_top5_similar_relations(relation_text, top_k=5):
    query_embedding = encoder_model.encode([relation_text])
    query_embedding = query_embedding.astype('float32')
    faiss.normalize_L2(query_embedding)
    
    distances, indices = index.search(query_embedding, top_k)
    
    top5_relations = []
    for idx in indices[0]:
        relation_label = wikidata_df.iloc[idx]['propertyLabel']
        top5_relations.append(relation_label)
    
    return top5_relations
def parse_relation_choices(output_text):
    mapping = {}
    pattern = r'Triple\s+(\d+):\s*(.+)'
    for match in re.findall(pattern, output_text):
        idx, relation = match
        mapping[int(idx)] = relation.strip()
    return mapping
def build_step2_prompt(description, triples_with_top5):
    triplets_text = ""
    for head, relation, tail, top5 in triples_with_top5:
        top5_str = ", ".join(top5)
        triplets_text += f"{head}\n{relation}: [{top5_str}]\n{tail}\n\n"
    
    prompt = f"""
    In the previous step, there were extracted triplets from the Wikidata knowledge graph. Each triplet contains two entities (subject and object) and one relation that connects these subject and object. However, some of the relations extracted in the previous step may have not an exact name from Wikidata.

    We linked each relation name with top similar exact names from the Wikidata by semantic similarity. Your task is to choose appropriate names for relations that correspond to the text's context and triplet they were taken from.

    Text: {description}

    Triplets and corresponding entity and relation mappings:
    {triplets_text}

    For each relation from the extracted triplets, you must choose the most appropriate name only from the corresponding list of 5 exact ones that better match each triplet and the context of the previously demonstrated text.

    You must ONLY output the final chosen relation for each triple, one per line, in this format:
    Triple 1: [chosen_relation]
    Triple 2: [chosen_relation]
    ...

    """
    
    return prompt

# ========== Step 1 Prompt Template ==========
STEP1_PROMPT = """
You are a structured knowledge extraction system capable of processing both text and images.
Your task is to extract meaningful triples in the format: [head], [relation], [tail], [source]
source indicates whether the information is derived from "text", "image".

Input:
A textual description of an entity (e.g., film, person, location)
One or more related images

Output Rules:
Each triple must be on its own line, separated by commas: [head], [relation], [tail], [source]  
Do not include any explanations, notes, or additional text outside the triples.
Do not use "Not specified", "Unknown", or similar placeholders as entities.
Distinguish whether the triple comes from an image or text accurately, Do not fabricate this information.
Do not generate entities including numerical attributes or time information. Such as 18794 square kilometers, 7000 people, 1973. etc
Do not generate repeat triples.
Try to generate around 5-10 quality and representative triples for one entity and Do not generate over 10 triples for one entity.

[Main Entity Priority]
The given main entity should preferably appear as the head or tail of the triples.
New entities can be introduced only if they can be reasonably inferred from the text, image, or general world knowledge.
Example: If the main entity is Washington, D.C. and the image shows a recognizable building that can be reasonably identified as the United States Capitol, you may introduce that entity and extract:
Washington, D.C., contains landmark, United States Capitol, image.

[Text First, Image as Supplement]
Text is the primary source for extracting triples.
Images are used to verify, supplement, or strengthen information inferred from the text.
The extraction should be a joint reasoning process combining both modalities.

[Entity Format Requirements]
Both head and tail entities must be clearly identifiable, well-named entities such as people, places, institutions, rivers, or established conceptual entities (e.g., American flag, government building).
Do not use Boolean or vague phrases, or numerical descriptions as entities (for example, “more than 20 million visitors annually” is invalid).
If a relation cannot be paired with a clearly identifiable, named entity from the provided text or image, DO NOT generate that triple. Never use placeholder phrases, explanatory notes, or bracketed comments as tail entities. Omitting a triple is preferred over fabricating or approximating an entity. 

[Textual and Visual Verifiability]
Do not fabricate entities or relations unsupported by the text or image.
Every extracted triple must be directly or indirectly supported by the text, the image, or logical inference grounded in both.
Example: If the image shows a building with a visible U.S. flag, this supports “has flag, American flag”.

[Clear and Coherent Relations]
Relations must be semantically clear and expressed as short words or phrases, not full sentences. Refer to the relations in the original dataset given below. 
When possible, reuse relations from the original dataset if they capture the intended meaning (for example, use “shares border with” instead of “borders”).
However, if an appropriate relation does not exist, you may create a new one that fits the meaning accurately.

[Category-Aware image Extraction Strategy]
When extracting triples based on images, follow the corresponding extraction strategy.(Only for image-based triples)

Category: LOCATION
Focus on landmarks, architecture, and natural features visible in the image.
Typical relations: located in, contains landmark, is part of, has flag, capital of, architectural style, shares border with.
New entities (such as United States Capitol) may be introduced if visually or textually supported.


Category: MOVIE POSTER
Focus on textual cues from the poster (title, director, actors, producer, distributor, etc.).
Typical relations: director, screenwriter, producer, distributor, cast member, composer, cinematography format.
New entities such as actor names or company names are allowed.


Category: PERSON (PORTRAIT)
Extract the person's representative features visible in the image.

Typical relations: wears, facial hair.
Example: J. D. Chakravarthy, wears, black suit, image.

Note: The examples and typical relations provided are only partial. You are encouraged to extract diverse relations but they need to adhere to the requirement of "[Clear and Coherent Relations]". For other relations not falling into the above categories, please use common sense and refer to the above categories to arrange the extraction strategy.

Relations in the original dataset：
adjacent station, allegiance, architect, architectural style, archives at, author, award received, based on, basic form of government, basin country, brand, candidate, capital, capital of, cast member, cause of death, chairperson, characters, child, child organization or unit, collection, color, composer, connecting line, contains settlement, contains the administrative territorial entity, continent, contributor to the creative work or subject, copyright license, country, country for sport, country of citizenship, country of origin, creator, depicts, designed by, developer, different from, elomatic relation, director, director of photography, discoverer or inventor, distributed by, distribution format, doctoral advisor, doctoral student, drafted by, educated at, employer, enclave within, ethnic group, executive producer, father, field of this occupation, field of work, film editor, filming location, followed by, follows, founded by, genre, given name, has part(s), has spin-off, head coach, headquarters location, industry, inflows, influenced by, inspired by, instance of, instrument, lake on watercourse, language of work or name, language regulatory body, language used, languages spoken, written or signed, league or competition, librettist, licensed to broadcast to, list of episodes, located in or next to body of water, located in the administrative territorial entity, located in time zone, located in/on physical feature, location, location of creation, location of formation, lyricist, main subject, maintained by, manufacturer, member of, member of political party, member of sports team, military branch, military or police rank, mother, mouth of the watercourse, movement, named after, narrative location, narrator, native language, notable work, occupant, occupation, official language, official residence, operating area, operating system, operator, opposite of, original broadcaster, original language of film or TV show, outflows, owned by, parent organization or unit, parent taxon, part of, part of the series, participant, participant in, participated in conflict, performer, place of birth, place of burial, place of death, platform, political alignment, political ideology, position played on team / speciality, present in work, producer, product or material produced, production company, production designer, programmed in, publisher, record label, religion or worldview, religious order, replaced by, replaces, residence, screenwriter, shares border with, shipping port, software engine, soundtrack release, sport, spouse, student, student of, subclass of, successful candidate, territory claimed by, time period, tracklist, twinned administrative body, unmarried partner, vessel class, voice actor, voice type, work location

# Now process the following entity and its description:
# Entity: {entity}
# Description: {description}
"""
refined_raw_path = "../outputs/ori_triples/MKG-W_triples_refined_raw1.txt"
f_refined = open(refined_raw_path, "w", encoding="utf-8")
all_triples = []

descriptions_dict = load_descriptions(description_file)



with open("../outputs/ori_triples/MKG-W_triples_src1.txt", "w", encoding="utf-8") as f:
    for idx, (entity_id, entity_name) in enumerate(entity_list[:1500]):
        try:
            print("=" * 80)
            print(f"📌 Entity: {entity_name}")
            
            if entity_name not in descriptions_dict:
                continue
            
            url, description = descriptions_dict[entity_name]
            
            image_paths = find_images_for_entity(entity_name, image_root)

            # image_path = find_images_for_entity(entity_name, image_root)

            print("\n🚀 STEP 1: triplet extraction...")
            user_content = []
            
            for img_path in image_paths:
                print(f"🖼️ image: {img_path}")
                encoded_img = encode_image_to_base64(img_path)
                user_content.append({
                    "type": "image_url",
                    "image_url": {"url": encoded_img},
                })

            prompt = STEP1_PROMPT.format(entity=entity_name, description=description)
            user_content.append({"type": "text", "text": prompt})

            messages = [
                {"role": "system", "content": "You are a structured knowledge extraction system."},
                {"role": "user", "content": user_content}
            ]

            step1_output = chat_with_model(model1, messages, temperature=0.1)

            triples_src = parse_triples_with_source(step1_output)
            f.write(f"### {entity_name} ###\n")
            for h, r, t, s in triples_src:
                f.write(f"{h}, {r}, {t}, {s}\n")
            triples = parse_triples(step1_output)
            
            if not triples:
                continue

            triples_with_top5 = []
            
            for head, relation, tail in triples:
                top5_relations = find_top5_similar_relations(relation)
                triples_with_top5.append((head, relation, tail, top5_relations))
                print(f"  relation '{relation}' -> Top5: {top5_relations}")
            
            # 构建Step 2 prompt
            step2_prompt = build_step2_prompt(description, triples_with_top5)
            
            messages_step2 = [
                {"role": "system", "content": "You are a knowledge graph relation refinement system."},
                {"role": "user", "content": step2_prompt}
            ]
            
            # completion_step2 = client.chat.completions.create(
            #     model=model2,
            #     messages=messages_step2,
            #     temperature=0.1,
            # )
            # selected_output = completion_step2.choices[0].message.content.strip()
            selected_output = chat_with_model(model2, messages_step2, temperature=0.1)
            print(selected_output)
            print("-" * 80)
            relation_mapping = parse_relation_choices(selected_output)
            refined_triples = []
            for i, (head, old_rel, tail, top5) in enumerate(triples_with_top5, 1):
                new_rel = relation_mapping.get(i, old_rel)  # 如果LLM没选，则保留原关系
                refined_triples.append((head, new_rel, tail))
                print(f"Triple {i}:")
                print(f"  Head: {head}")
                print(f"  Tail: {tail}")
                print(f"  origin relation: {old_rel}")
                print(f"  Top5 : {top5}")
                print(f"  ✅ choice: {new_rel}")
                print("-" * 60)


            f_refined.write(f"### {entity_name} ###\n")
            for head, relation, tail in refined_triples:
                f_refined.write(f"{head}, {relation}, {tail}\n")
                all_triples.append((head, relation, tail))

            
        except Exception as e:
            print(f"error: {e}")
            continue

clean_set = set()
removed_not_specified = 0

for head, rel, tail in all_triples:
    tl = tail.lower()
    if "[not specified" in tl or "(not specified)" in tl or "[none specified" in tl:
        removed_not_specified += 1
        continue
    clean_set.add((head, rel, tail))



os.makedirs(os.path.dirname(output_file), exist_ok=True)

with open(output_file, "w", encoding="utf-8") as fout:
    for h, r, t in sorted(clean_set):
        fout.write(f"{h}, {r}, {t}\n")




print("\n🏁 complete！")