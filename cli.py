from utils import load_verses
from search_engine import (
    build_tfidf_index, search_verses,
    build_semantic_index, semantic_search
)

def main():
    print("🕌 Quranic Verse Search Engine")
    print("🔎 Type a search query to find matching verses.")
    print("✏️  Type 'exit' to quit the program.\n")

    # Load data
    verses = load_verses()
    if not verses:
        print("⚠️ Failed to load Quran data.")
        return

    # Choose mode
    mode = ""
    while mode not in ["1", "2"]:
        print("Choose search mode:")
        print("1️⃣  TF-IDF (Keyword Search)")
        print("2️⃣  Semantic AI Search")
        mode = input("Enter 1 or 2: ").strip()

    if mode == "1":
        vectorizer, tfidf_matrix = build_tfidf_index(verses)
    else:
        model, embeddings = build_semantic_index(verses)

    # Start search loop
    while True:
        query = input("\n🔍 Enter your search query: ").strip()
        if query.lower() == "exit":
            print("👋 Exiting. Goodbye!")
            break

        if mode == "1":
            results = search_verses(query, verses, vectorizer, tfidf_matrix)
        else:
            results = semantic_search(query, verses, model, embeddings)

        if not results:
            print("❌ No matching verses found.")
        else:
            print(f"\n✅ Found {len(results)} matching verse(s):")
            for verse, score in results:
                print(f"\n📖 Surah: {verse['surah']} | Ayah: {verse['ayah_number']}")
                print(f"   Arabic : {verse['text']}")
                print(f"   English: {verse['english']}")
                print(f"   Urdu   : {verse['urdu']}")
                print(f"   🔹 Relevance Score: {score:.2f}")

if __name__ == "__main__":
    main()
