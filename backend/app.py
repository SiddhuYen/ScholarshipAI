from flask import Flask, request, jsonify
from flask_cors import CORS
from openai import OpenAI
from dotenv import load_dotenv
import pandas as pd
import numpy as np
import json
import os
import boto3

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = Flask(__name__)
CORS(app)

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

SCHOLARSHIP_CSV = os.path.join(DATA_DIR, "scholarships_parsed.csv")
SCHOLARSHIP_EMB_NPY = os.path.join(DATA_DIR, "scholarship_embeddings.npy")
ESSAY_CSV = "essay_table_clean.csv"
ESSAY_TREE = "essay_tree.json"

MIN_ESSAY_WORDS = 150
ADVICE_LIMIT = 5
TOP_K = 50


def file_exists_and_nonempty(path):
    return os.path.exists(path) and os.path.getsize(path) > 0


def word_count(text):
    return len(str(text).split())


def download_from_r2_if_needed():
    remote_to_local = {
        "scholarships_parsed.csv": SCHOLARSHIP_CSV,
        "scholarship_embeddings.npy": SCHOLARSHIP_EMB_NPY,
    }

    missing = [local for local in remote_to_local.values() if not file_exists_and_nonempty(local)]
    if not missing:
        return

    account_id = os.getenv("R2_ACCOUNT_ID")
    access_key = os.getenv("R2_ACCESS_KEY_ID")
    secret_key = os.getenv("R2_SECRET_ACCESS_KEY")
    bucket = os.getenv("R2_BUCKET")

    if not all([account_id, access_key, secret_key, bucket]):
        raise RuntimeError(
            "Missing R2 environment variables. Required: "
            "R2_ACCOUNT_ID, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY, R2_BUCKET"
        )

    endpoint_url = f"https://{account_id}.r2.cloudflarestorage.com"

    s3 = boto3.client(
        "s3",
        endpoint_url=endpoint_url,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name="auto",
    )

    for remote_name, local_path in remote_to_local.items():
        if not file_exists_and_nonempty(local_path):
            print(f"Downloading {remote_name} from R2...")
            s3.download_file(bucket, remote_name, local_path)
            print(f"Downloaded {remote_name} -> {local_path}")


def ensure_required_local_files():
    required = [SCHOLARSHIP_CSV, SCHOLARSHIP_EMB_NPY, ESSAY_CSV, ESSAY_TREE]
    missing = [path for path in required if not file_exists_and_nonempty(path)]
    if missing:
        raise FileNotFoundError(f"Missing required files: {missing}")


def normalize_rows(matrix):
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return matrix / norms


def normalize_vector(vec):
    norm = np.linalg.norm(vec)
    if norm == 0:
        return vec
    return vec / norm


download_from_r2_if_needed()
ensure_required_local_files()

scholarship_df = pd.read_csv(SCHOLARSHIP_CSV)
essay_df = pd.read_csv(ESSAY_CSV)
scholarship_embeddings = np.load(SCHOLARSHIP_EMB_NPY)

with open(ESSAY_TREE, "r") as f:
    tree = json.load(f)

if len(scholarship_df) != len(scholarship_embeddings):
    raise ValueError(
        f"Mismatch: scholarship_df has {len(scholarship_df)} rows but "
        f"scholarship_embeddings has {len(scholarship_embeddings)} vectors"
    )

scholarship_embeddings = normalize_rows(scholarship_embeddings)


def embed(text):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=[str(text)[:8000]]
    )
    return np.array(response.data[0].embedding, dtype=np.float32)


def embed_batch(texts):
    cleaned = [str(text)[:8000] for text in texts]
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=cleaned
    )
    return [
        np.array(item.embedding, dtype=np.float32)
        for item in response.data
    ]


def cosine_similarity(a, b):
    a = np.array(a, dtype=np.float32)
    b = np.array(b, dtype=np.float32)

    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)

    if a_norm == 0 or b_norm == 0:
        return 0.0

    return float(np.dot(a, b) / (a_norm * b_norm))


def match_prompt(prompt, node):
    if not node["children"]:
        return node["essays"]

    options = ""
    for i, child in enumerate(node["children"]):
        sample_indices = child["essays"][:2]
        samples = " | ".join(
            [str(essay_df.loc[j, "Prompt"])[:100] for j in sample_indices if j < len(essay_df)]
        )
        options += f"{i}: {child['name']} (e.g. {samples})\n\n"

    response = client.chat.completions.create(
        model="gpt-4o",
        max_tokens=100,
        messages=[
            {
                "role": "system",
                "content": """You are a precise classifier. Return only a JSON object with a single key 'choice'.
Always pick the closest matching cluster even if the match isn't perfect.
Only return -1 if the prompt is completely unrelated to all clusters.
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

    if not isinstance(choice, int) or choice < 0 or choice >= len(node["children"]):
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


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "ok": True,
        "scholarships_loaded": len(scholarship_df),
        "embeddings_shape": list(scholarship_embeddings.shape),
        "essay_rows": len(essay_df)
    })


@app.route("/match", methods=["POST"])
def match():
    data = request.get_json(silent=True) or {}

    user_profile = data.get("profile", {})
    user_essays = data.get("essays", [])

    if not isinstance(user_essays, list) or len(user_essays) == 0:
        return jsonify({"error": "Request must include a non-empty 'essays' list."}), 400

    filtered_essays = []
    for essay in user_essays:
        prompt = str(essay.get("prompt", "")).strip()
        response_text = str(essay.get("response", "")).strip()

        if prompt and response_text and word_count(response_text) >= MIN_ESSAY_WORDS:
            filtered_essays.append({
                "prompt": prompt,
                "response": response_text
            })

    if len(filtered_essays) == 0:
        return jsonify({
            "error": f"No essays were at least {MIN_ESSAY_WORDS} words long."
        }), 400

    user_essays = filtered_essays

    essay_texts = []
    for essay in user_essays:
        prompt = essay["prompt"]
        response_text = essay["response"]
        text = f"{user_profile}\n{prompt}\n{response_text}".strip()
        essay_texts.append(text)

    batched_embeddings = embed_batch(essay_texts)

    essay_embeddings = {
        i: normalize_vector(emb)
        for i, emb in enumerate(batched_embeddings)
    }

    user_vecs = np.array(list(essay_embeddings.values()), dtype=np.float32)
    user_profile_vec = normalize_vector(np.mean(user_vecs, axis=0))

    scores_array = scholarship_embeddings @ user_profile_vec
    top_indices = np.argsort(scores_array)[::-1][:TOP_K]

    results = []

    for idx in top_indices:
        score = float(scores_array[idx])
        row = scholarship_df.iloc[idx]
        scholarship_emb = scholarship_embeddings[idx]

        if not essay_embeddings:
            continue

        best_essay_idx = max(
            essay_embeddings.keys(),
            key=lambda i: float(np.dot(essay_embeddings[i], scholarship_emb))
        )

        if best_essay_idx >= len(user_essays):
            continue
        
        best_essay = user_essays[best_essay_idx]

        advice = ""
        if len(results) < ADVICE_LIMIT:
            advice = generate_adaptation_advice(
                str(row.get("Purpose", "")),
                str(best_essay.get("prompt", "")),
                str(best_essay.get("response", ""))
            )

        results.append({
            "scholarship_url": row.get("url", ""),
            "scholarship_purpose": row.get("Purpose", ""),
            "match_score": round(score, 3),
            "best_essay_prompt": best_essay.get("prompt", ""),
            "best_essay_response": best_essay.get("response", ""),
            "adaptation_advice": advice
        })

    return jsonify(results)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)