# 📖 AL-BAYAN | AI-Powered Quranic Verse Search Engine
> *Bridging Traditional Keyword Search with Semantic AI Understanding*

**AL-BAYAN** is a cutting-edge, AI-driven Quranic search engine developed as a Final Year Project (FYP). Designed to transcend the limitations of rigid exact-match searches, it leverages **Hybrid Search** (`TF-IDF` + `Semantic Embeddings`) and **Retrieval-Augmented Generation (RAG)** to deliver highly accurate, context-aware, and scholarly Quranic verse retrieval. Whether you're a student, researcher, or lifelong learner, AL-BAYAN provides an intuitive, multi-lingual platform to explore the Quran with depth, clarity, and reverence.

---

## 🎯 Project Objectives
- 🔍 **Semantic Understanding:** Move beyond keyword matching to grasp contextual and thematic relevance.
- 🌍 **Multi-Language Support:** Seamless search across Arabic, English (Sahih International), and Urdu.
- 🤖 **AI-Augmented Insights:** Generate contextual explanations & scholarly summaries using Google Gemini 1.5 Flash.
- 💻 **Modern Accessibility:** Deliver a responsive, user-friendly interface optimized for study and research.

---

## ✨ Key Features
| Feature | Description |
|:---|:---|
| 🔄 **Hybrid Search Engine** | Combines `TF-IDF` for precise lexical matching with `Sentence Transformers` (`all-MiniLM-L6-v2`) for deep semantic similarity. |
| 🧠 **AI-Powered Insights (RAG)** | Retrieves top-k verses and feeds them into an LLM pipeline to generate contextual, tafsir-backed explanations. |
| 🌐 **Tri-Lingual Support** | Full Quranic text with parallel English & Urdu translations, plus Tafsir Ibn Kathir in both languages. |
| 🎙️ **Voice-Activated Search** | Hands-free verse discovery using modern Web Speech API integration. |
| ⚙️ **Smart UI Controls** | Toggle dark/light mode, adjust font scaling, and switch translations dynamically without page reloads. |
| 📤 **Shareable Verse Cards** | Generate beautifully formatted, social-media-ready verse images for easy sharing & da'wah. |

---

## 🛠️ Technology Stack
| Category | Technologies & Libraries |
|:---|:---|
| **Backend** | `Python`, `Flask` |
| **AI / Machine Learning** | `PyTorch`, `Sentence-Transformers` (`all-MiniLM-L6-v2`), `Scikit-learn` (TF-IDF) |
| **LLM Integration** | `Google GenAI SDK` (Gemini 1.5 Flash) |
| **Frontend** | `HTML5`, `Tailwind CSS`, `Vanilla JavaScript` |
| **Data Pipeline** | Custom Quran JSON Dataset, Tafsir Ibn Kathir (EN/UR), Runtime Dataset Merging |

---

## 🔍 How It Works (Under the Hood)
1. **Query Normalization:** User input (text or voice) is cleaned, tokenized, and language-detected.
2. **Dual-Path Retrieval:**
   - 📊 `TF-IDF` ranks verses based on exact keyword frequency & inverse document frequency.
   - 🌐 `Sentence Transformers` compute cosine similarity between query and verse embeddings.
   - ⚖️ Results are fused using a weighted hybrid scoring algorithm for optimal precision & recall.
3. **RAG Generation:** Top-matching verses + relevant Tafsir excerpts are passed to **Gemini 1.5 Flash** with a structured prompt to generate scholarly, context-aware summaries.
4. **Dynamic Rendering:** Results are injected into the UI with translation toggles, font scaling, and export/share functionality.

---

## 📂 Project Structure
```
AL-BAYAN/
│
├── app.py                      # Main Flask server & routing
├── search_engine.py            # Hybrid search logic & RAG pipeline
├── models.py                   # ML model initialization & embedding generation
├── utils.py                    # Utility & helper functions
├── cli.py                      # Optional command-line interface
├── requirements.txt            # Python dependencies
├── .gitignore                  # Version control exclusions
│
├── templates/                  # Frontend HTML templates (Jinja2)
│   ├── base.html               # Base layout & navigation
│   ├── index.html              # Homepage & search entry
│   ├── search.html             # Results & AI insights display
│   ├── browse.html             # Surah/Ayah navigator
│   └── about.html              # Project documentation
│
├── static/                     # Frontend assets
│   ├── css/styles.css          # Custom styling overrides
│   ├── js/main.js              # Client-side interactivity
│   └── assets/                 # Screenshots & media
│
├── data/                       # Quranic & Tafsir datasets
│   ├── quran_part_1.json       # Surah 1–57
│   ├── quran_part_2.json       # Surah 58–114
│   └── sources/                # Raw/backup source files
│
└── scripts/                    # Data preprocessing & embedding scripts
    ├── merge_english_urdu.py   # Bilingual dataset merger
    ├── merge_tafseer.py        # Tafsir integration script
    ├── final_merge.py          # Final dataset consolidation
    └── precompute_embeddings.py # Vector embedding generator
```
> 💡 *Note: To comply with GitHub’s 25 MB file size limit, the final Quranic dataset is split into two JSON files and dynamically merged at runtime.*

---

## 🚀 Installation & Setup
Follow these steps to run AL-BAYAN locally:

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/AL-BAYAN.git
cd AL-BAYAN
```

### 2️⃣ Create & Activate Virtual Environment (Recommended)
```bash
python -m venv venv
source venv/bin/activate   # macOS/Linux
venv\Scripts\activate      # Windows
```

### 3️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 4️⃣ Configure Google Gemini API Key
- Obtain your API key from [Google AI Studio](https://aistudio.google.com/).
- Set it as an environment variable:
  ```bash
  export GEMINI_API_KEY="your_api_key_here"  # macOS/Linux
  set GEMINI_API_KEY=your_api_key_here       # Windows CMD
  $env:GEMINI_API_KEY="your_api_key_here"    # PowerShell
  ```
- *(Optional for local testing: You may hardcode the key in `app.py`, but environment variables are strongly recommended for security.)*

### 5️⃣ Run the Application
```bash
python app.py
```
🌐 Open your browser and visit: `http://127.0.0.1:5000`

---

## 📸 Screenshots & UI Preview
*(Add your UI captures to `static/assets/` and update the paths below)*

| 🏠 Home | 🔍 Search & AI Insights | 📖 Browse Surah | ℹ️ About |
|:---:|:---:|:---:|:---:|
| `![Home](static/assets/Home.png)` | `![Search](static/assets/Search.png)` | `![Browse](static/assets/Browse_Surah.png)` | `![About](static/assets/About.png)` |

---

## 📜 License & Academic Use
This project is developed strictly for **educational and academic purposes** as a Final Year Project (FYP). It is not intended for commercial distribution, fatwa issuance, or religious authority. All Quranic text, translations, and Tafsir content are used with the utmost respect and solely for research, learning, and technological exploration.

---

## 🕌 Acknowledgments
- 📖 The Holy Quran & its respected translators (Sahih International, Urdu scholarly translations)
- 📚 Tafsir Ibn Kathir & classical Islamic scholarship
- 🤖 Open-source AI/ML community: Hugging Face, Google GenAI, Scikit-learn, Flask, Tailwind CSS
- 🎓 Faculty advisors, peer reviewers, and academic supporters who guided this FYP

---

## 🤝 Contributing & Feedback
As an academic project, this repository is primarily for demonstration, documentation, and educational sharing. However, constructive feedback, bug reports, or academic collaborations are always welcome. Please open an issue or submit a pull request following standard GitHub practices.

---
*Built with ❤️ and reverence for knowledge.*  
**AL-BAYAN** – *Where AI meets Divine Wisdom.*

---
