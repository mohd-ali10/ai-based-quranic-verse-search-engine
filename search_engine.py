from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, util

# === TF-IDF SEARCH ===

def build_tfidf_index(verses):
    """
    Build TF-IDF index from English translations of verses.
    """
    texts = [verse.get("english", "") for verse in verses]
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(texts)
    return vectorizer, tfidf_matrix

def search_verses(query, verses, vectorizer, tfidf_matrix, top_k=5):
    """
    Perform TF-IDF search on verses and return top-k results.
    """
    query_vec = vectorizer.transform([query])
    cosine_similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()
    top_indices = cosine_similarities.argsort()[::-1][:top_k]

    results = []
    for idx in top_indices:
        score = cosine_similarities[idx]
        if score > 0:
            verse = verses[idx]
            results.append((verse, score))
    return results

# === SEMANTIC SEARCH ===

def build_semantic_index(verses, model_name="all-MiniLM-L6-v2"):
    """
    Build sentence-transformer embeddings for all verses.
    """
    model = SentenceTransformer(model_name)
    texts = [verse.get("english", "") for verse in verses]
    embeddings = model.encode(texts, convert_to_tensor=True)
    return model, embeddings

def semantic_search(query, verses, model, embeddings, top_k=5):
    """
    Perform semantic search using cosine similarity.
    """
    query_embedding = model.encode(query, convert_to_tensor=True)
    hits = util.semantic_search(query_embedding, embeddings, top_k=top_k)[0]

    results = []
    for hit in hits:
        verse = verses[hit["corpus_id"]]
        score = hit["score"]
        results.append((verse, score))
    return results
