from flask import Flask, render_template, request, jsonify
import json  # <--- Added import
from utils import load_verses
from search_engine import (
    build_tfidf_index,
    search_verses,
    build_semantic_index,
    semantic_search
)
from rag import maybe_llm_answer, make_extractive_answer

app = Flask(__name__)

# --- 1. Load Data ---

# Load Quranic verses
verses = load_verses()

# Load Tafsir Data (New)
print("⏳ Loading Tafsir data...")
try:
    with open('tafsir_ibn_kathir_eng_ur.json', 'r', encoding='utf-8') as f:
        tafsir_data = json.load(f)
    print("✅ Tafsir data loaded.")
except FileNotFoundError:
    print("⚠️ Tafsir file not found. Tafsir features will be disabled.")
    tafsir_data = {}

# --- 2. Build Search Indices ---

# Build TF-IDF index
vectorizer, tfidf_matrix = build_tfidf_index(verses)

# Build Semantic index
semantic_model, semantic_embeddings = build_semantic_index(verses)

# --- 3. Routes ---

@app.route('/', methods=['GET', 'POST'])
def index():
    results = []
    query = ''
    mode = 'tfidf'  # default mode

    if request.method == 'POST':
        query = request.form.get('query', '').strip()
        mode = request.form.get('mode', 'tfidf')

        if query:
            if mode == 'semantic':
                results = semantic_search(query, verses, semantic_model, semantic_embeddings)
            else:
                results = search_verses(query, verses, vectorizer, tfidf_matrix)

    return render_template('index.html', query=query, results=results, mode=mode)

# --- New Route: Get Tafsir ---
@app.route('/get_tafsir/<int:surah>/<int:ayah>')
def get_tafsir(surah, ayah):
    # The JSON keys are formatted as "Surah:Ayah" (e.g., "1:1")
    key = f"{surah}:{ayah}"
    data = tafsir_data.get(key)
    
    if data:
        return jsonify(data)
    else:
        return jsonify({
            "en": "<p>Tafsir not available for this verse.</p>", 
            "ur": "<p>اس آیت کی تفسیر دستیاب نہیں ہے۔</p>"
        }), 404

# --- Chat Routes (Kept for later use) ---
@app.route('/chat-ui', methods=['GET'])
def chat_ui():
    return render_template('chat.html')

@app.route('/chat', methods=['POST'])
def chat():
    payload = request.get_json(force=True) or {}
    question = (payload.get("question") or "").strip()
    if not question:
        return jsonify({"answer": "Please type a question.", "verses": []})

    top_k = payload.get("top_k", 5)

    # Use semantic search for retrieval
    results = semantic_search(question, verses, semantic_model, semantic_embeddings, top_k=top_k)
    top_verses = [v for v, _ in results]

    # Try LLM answer if enabled, otherwise use extractive
    llm_ans = maybe_llm_answer(question, top_verses)
    answer = llm_ans if llm_ans else make_extractive_answer(question, top_verses)

    return jsonify({
        "answer": answer,
        "verses": [
            {
                "surah": v.get("surah"),
                "ayah_number": v.get("ayah_number"),
                "arabic": v.get("text"),
                "english": v.get("english"),
                "urdu": v.get("urdu")
            } for v in top_verses
        ]
    })

if __name__ == '__main__':
    app.run(debug=True)