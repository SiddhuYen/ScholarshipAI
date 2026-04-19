from flask import Flask, request, jsonify
from flask_cors import CORS
from openai import OpenAI
from dotenv import load_dotenv
import pandas as pd
import numpy as np
import json
import os

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = Flask(__name__)
CORS(app)

# Load everything once at startup
scholarship_df = pd.read_csv("scholarships_parsed.csv")
essay_df = pd.read_csv("essay_table_clean.csv")

with open("scholarship_embeddings.json", "r") as f:
    scholarship_embeddings = json.load(f)

with open("essay_tree.json", "r") as f:
    tree = json.load(f)

def embed(text):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=str(text)[:8000]
    )
    return response.data[0].embedding

def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def match_prompt(prompt, node):
    if not node["children"]:
        return node["essays"]
    
    options = ""
    for i, child in enumerate(node["children"]):
        sample_indices = child["essays"][:2]
        samples = " | ".join([str(essay_df.loc[j, "Prompt"])[:100] for j in sample_indices])
        options += f"{i}: {child['name']} (e.g. {samples})\n\n"
    
    response = client.chat.completions.create(
        model="gpt-4o",
        max_tokens=100,
        messages=[
            {
                "role": "system",
                "content": """You are a precise classifier. Return only a JSON object with a single key 'choice'.
Always pick the closest matching cluster even if the match isn't perfect.
Only return -1 if the prompt is completely unrelated to all clusters (e.g. ministry, music therapy, environmental activism).
When in doubt, pick the closest cluster."""
            },
            {
                "role": "user",
                "content": f"""A student is applying for a scholarship with this prompt:
"{prompt}"

Which cluster contains essays that would BEST answer this prompt?
Pick the most specific match, not the most general one.

Clusters:
{options}

Return only: {{"choice": <number or -1>}}"""
            }
        ]
    )
    
    raw = response.choices[0].message.content.strip()
    raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    result = json.loads(raw)
    choice = result["choice"]
    
    if choice == -1:
        return []
    
    return match_prompt(prompt, node["children"][choice])

def generate_adaptation_advice(scholarship_purpose, essay_prompt, essay_response):
    response = client.chat.completions.create(
        model="gpt-4o",
        max_tokens=300,
        messages=[{
            "role": "user",
            "content": f"""A student is adapting their existing essay for a scholarship.

Scholarship purpose: {scholarship_purpose}
Their existing essay prompt: {essay_prompt}
Their existing essay (first 500 chars): {essay_response[:500]}

In 3 bullet points, tell them specifically what to change to make this essay fit the scholarship. Be concrete, not generic."""
        }]
    )
    return response.choices[0].message.content

@app.route("/match", methods=["POST"])
def match():
    data = request.json
    user_profile = data["profile"]
    user_essays = data["essays"]  # list of {prompt, response} dicts

    # Embed user essays
    essay_embeddings = {}
    for i, essay in enumerate(user_essays):
        text = essay["prompt"] + " " + essay["response"]
        essay_embeddings[i] = embed(text)

    # Average all essay embeddings as user profile vector
    user_vecs = list(essay_embeddings.values())
    user_profile_vec = np.mean(user_vecs, axis=0)

    # Find top scholarships via cosine similarity
    scores = []
    for idx, emb in scholarship_embeddings.items():
        score = cosine_similarity(user_profile_vec, emb)
        scores.append((int(idx), score))
    scores.sort(key=lambda x: x[1], reverse=True)
    top_50 = scores[:50]

    # Match essays to each scholarship
    results = []
    for idx, score in top_50:
        row = scholarship_df.iloc[idx]
        scholarship_text = f"{row['Purpose']} {row['Criteria']}"

        matched_essay_indices = match_prompt(scholarship_text, tree)
        if not matched_essay_indices:
            continue

        scholarship_emb = scholarship_embeddings.get(str(idx))
        if not scholarship_emb:
            continue

        best_essay_idx = max(
            matched_essay_indices,
            key=lambda i: cosine_similarity(
                essay_embeddings[i],
                scholarship_emb
            ) if i in essay_embeddings else 0
        )

        best_essay = user_essays[best_essay_idx]
        advice = generate_adaptation_advice(
            row["Purpose"],
            best_essay["prompt"],
            best_essay["response"]
        )

        results.append({
            "scholarship_url": row["url"],
            "scholarship_purpose": row["Purpose"],
            "match_score": round(score, 3),
            "best_essay_prompt": best_essay["prompt"],
            "best_essay_response": best_essay["response"],
            "adaptation_advice": advice
        })

    return jsonify(results)

if __name__ == "__main__":
    app.run(debug=True, port=5000)