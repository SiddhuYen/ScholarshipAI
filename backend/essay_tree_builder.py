from openai import OpenAI
from dotenv import load_dotenv
import pandas as pd
import json
import os

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

df = pd.read_csv("essay_table_clean.csv")

def cluster_with_gpt(indices, n_clusters=3):
    essays_text = ""
    for i in indices:
        row = df.loc[i]
        essays_text += f"Index {i}: {row['Prompt']}\n\n"
    
    max_retries = 5
    for attempt in range(max_retries):
        response = client.chat.completions.create(
            model="gpt-4o",
            max_tokens=2000,
            messages=[
                {
                    "role": "system",
                    "content": """You are a precise JSON generator. 
You must include EVERY index provided. 
Never duplicate an index.
Never omit an index.
Return only raw JSON, no markdown, no explanation."""
                },
                {
                    "role": "user",
                    "content": f"""Group these {len(indices)} essay indices into exactly {n_clusters} thematic clusters.

RULES:
- Every index listed below must appear in exactly one cluster
- No index may appear twice
- Do not add indices that aren't listed
- Return only raw JSON

Required indices: {indices}

Essays:
{essays_text}

Return this exact format:
{{
    "clusters": [
        {{
            "name": "descriptive cluster name",
            "indices": [index numbers here]
        }}
    ]
}}"""
                }
            ]
        )
        
        raw = response.choices[0].message.content.strip()
        raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        
        try:
            clusters = json.loads(raw)["clusters"]
            all_assigned = [i for cluster in clusters for i in cluster["indices"]]
            
            duplicates = [i for i in all_assigned if all_assigned.count(i) > 1]
            missing = set(indices) - set(all_assigned)
            extra = set(all_assigned) - set(indices)
            
            if duplicates or missing or extra:
                print(f"Attempt {attempt+1} failed: duplicates={duplicates}, missing={missing}, extra={extra}")
                continue
            
            print(f"Attempt {attempt+1} succeeded with {len(clusters)} clusters")
            return clusters
            
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Attempt {attempt+1} JSON error: {e}")
            continue
    
    # Manual fallback
    print("GPT failed, splitting manually")
    chunk_size = max(1, len(indices) // n_clusters)
    return [
        {"name": f"Group {i+1}", "indices": indices[i*chunk_size:(i+1)*chunk_size]}
        for i in range(n_clusters)
    ]

def build_tree(indices, depth=0):
    print(f"{'  ' * depth}Building node with {len(indices)} essays at depth {depth}")
    
    if len(indices) <= 5:
        return {
            "name": "Leaf",
            "essays": indices,
            "children": []
        }
    
    n_clusters = 3 if len(indices) <= 20 else 4
    clusters = cluster_with_gpt(indices, n_clusters)
    
    children = []
    for cluster in clusters:
        child = build_tree(cluster["indices"], depth + 1)
        child["name"] = cluster["name"]
        children.append(child)
    
    return {
        "name": "Node",
        "essays": indices,
        "children": children
    }

# Build root level from existing Topic Cluster column
root_clusters = df.groupby("Topic Cluster").apply(
    lambda x: x.index.tolist()
).to_dict()

print("Root clusters:")
for name, indices in root_clusters.items():
    print(f"  {name}: {len(indices)} essays")

# Build children for each root cluster
children = []
for cluster_name, indices in root_clusters.items():
    child = build_tree(indices)
    child["name"] = cluster_name
    children.append(child)

# Assemble full tree
tree = {
    "name": "Root",
    "essays": list(df.index),
    "children": children
}

with open("essay_tree.json", "w") as f:
    json.dump(tree, f, indent=2)

print("\nDone. Tree saved to essay_tree.json")