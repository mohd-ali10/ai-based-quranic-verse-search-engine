import json

class Verse:
    def __init__(self, surah, ayah_number, english, urdu, text=""):
        self.surah = surah
        self.ayah_number = ayah_number
        self.english = english
        self.urdu = urdu
        self.text = text  # Arabic verse



def load_quran_data(filepath):
    """
    Loads Quranic verses from a JSON file and returns a list of Verse objects.
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)

    verses = []
    for item in raw_data:
        verses.append(
            Verse(
                surah=item.get("surah", ""),
                ayah_number=item.get("ayah_number", ""),
                english=item.get("english", ""),
                urdu=item.get("urdu", "")
            )
        )
    return verses
