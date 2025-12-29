import json

def load_verses(filepath="data/quran_with_urdu.json"):
    """
    Load and flatten Quranic verses from the merged dataset.
    Each verse will include: Surah name, verse number, Arabic text, English translation, and Urdu translation.
    """
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"❌ File not found: {filepath}")
        return []

    verses = []

    for surah in data:
        surah_name = surah.get("transliteration", surah.get("name", ""))
        for verse in surah["verses"]:
            verses.append({
    "surah": surah["transliteration"],
    "ayah_number": verse["id"],
    "text": verse["text"],             # Arabic
    "english": verse["english"],       # English
    "urdu": verse["urdu"]              # Urdu
})


    return verses
