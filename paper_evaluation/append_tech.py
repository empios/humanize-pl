from pathlib import Path

file_path = Path("paper_evaluation/baseline_results.md")
content = file_path.read_text(encoding="utf-8")

tech_section = """
## Metodologia i Użyte Narzędzia (Tech Stack)

Aby zapewnić rzetelność i powtarzalność eksperymentu, poniżej zestawiono narzędzia wykorzystane w obu podejściach:

### 1. Podejście "Bare LLM" (Zero-Shot Baseline)
*   **Model językowy:** Gemini 1.5 Pro (uruchomiony w środowisku agentowym w trybie zero-shot).
*   **Strategia:** *Zero-shot prompting* bez dostępu do zewnętrznych narzędzi i bez narzuconych reguł ("blank state").
*   **Charakterystyka:** Model opierał się wyłącznie na autoregresyjnym przewidywaniu tokenów bazując na swoich wagach wewnętrznych. Brak mechanizmów samokontroli i deterministycznej weryfikacji poprawności logicznej względem polskiego prawa.

### 2. Silnik "Humanize-PL" (Podejście Neuro-Symboliczne / Hybrid)
Silnik wykorzystuje kaskadę walidacyjną, łączącą deterministyczne reguły lingwistyczne z precyzyjną weryfikacją na bazie wąskich sieci neuronowych:
*   **Analiza Składniowa i Morfologiczna (Symbolic):**
    *   **Stanza (Stanford NLP):** Zaawansowane parsery zależnościowe (dependency trees), tokenizacja oraz tagowanie POS (Part-of-Speech) dla języka polskiego, niezbędne m.in. do rozbijania skomplikowanych łańcuchów dopełniaczowych.
    *   **Morfeusz 2:** Zaawansowany analizator morfologiczny dedykowany dla języka polskiego, gwarantujący dokładną detekcję form gramatycznych (np. wolicjonalnych i imperatywnych) na potrzeby strażnika profili deontycznych.
    *   **Autorskie algorytmy lingwistyczne:** Obliczanie metryk bogactwa językowego (MTLD / HD-D) oraz rytmiki i wariancji tekstu (Burstiness & Sentence Length Entropy) zoptymalizowane pod detekcję tzw. "AI-artifacts".
*   **Neuronowe Bramki Zabezpieczające (Neural Gates):**
    *   **Strażnik Płynności (Fluency Scorer):** Językowy model **`allegro/herbert-base-cased`** (polski BERT), wykorzystywany do oceny maskowanej (Masked LM) w celu odrzucania przekształceń pogarszających naturalność i "flow" zdania.
    *   **Strażnik Sensu (Semantic Entailment / NLI):** Wielojęzyczny cross-encoder **`MoritzLaurer/mDeBERTa-v3-base-mnli-xnli`**. Pełni rolę bezwzględnej "bramki logicznej" wykrywającej sprzeczności (*contradiction* i *neutral*) pomiędzy oryginałem a wersją przeredagowaną, blokując halucynacje prawne na poziomie semantycznym.
"""

if "Metodologia i Użyte Narzędzia" not in content:
    file_path.write_text(content + "\n" + tech_section, encoding="utf-8")
    print("Dodano pomyślnie.")
else:
    print("Sekcja już istnieje.")
