"""Synthetic regression fixtures. NOT an evaluation corpus.

Read this before using anything in paper_evaluation/dataset/ for a measurement.

Every slot below is a family the detector already looks for: `openings` is
discourse_frame, `empty_emphasis` is abstract_frame, `typographical` is the
em-dash artifact, `connectors` is transition_marker. Documents assembled from
these lists contain the patterns by construction, so scoring the detector on
them measures nothing: it asks whether the engine finds what we planted.

It also produces ungrammatical Polish - "Kwestia ta jest niezbedne dla
zapewnienia", "Oraz Umowa wchodzi w zycie" - because the connector is glued
onto a capitalised core clause. Part of what the engine appears to "fix" on
this dataset is cleanup after this generator.

These files remain useful as deterministic fixtures for rule regressions,
where planting the pattern is the point. For evaluation use
evaluate_engine.py, which runs on real model output from
docs_tests/ai_generated/ and separates it by provenance.
"""

import os
import random

os.makedirs("paper_evaluation/dataset", exist_ok=True)

openings = [
    "Warto wskazać, że ", "Należy zauważyć, że ", "Ważnym jest, aby podkreślić, iż ",
    "Zasadnym jest wskazanie, że ", "Istotnym jest, że ", "Należy zważyć, że "
]

nominalizations = [
    "celem zapewnienia możliwości realizacji dostępu do usług",
    "w celu dokonania weryfikacji poprawności wykonania zobowiązania",
    "dla zapewnienia prawidłowości procesu akceptacji",
    "w kontekście dokonania zapłaty należności",
    "w ramach świadczenia usług wsparcia technicznego",
    "celem uniknięcia nałożenia kary finansowej"
]

typographical = [
    "—wbrew twierdzeniom stron—", "—co do zasady—", "—w świetle orzecznictwa—",
    "—w przypadku braku zgody—", "—zgodnie z literaturą—", "—w opinii organów—"
]

empty_emphasis = [
    "odgrywa kluczową rolę dla funkcjonowania",
    "ma istotne znaczenie w kontekście",
    "stanowi fundament dla zabezpieczenia",
    "jest niezbędne dla zapewnienia bezpieczeństwa"
]

balanced_pairs = [
    "Z jednej strony powód wnosi o zapłatę, z drugiej zaś strony pozwany uchyla się od odpowiedzialności.",
    "Z jednej strony umowa chroni interesy Zamawiającego, z drugiej strony narzuca rygorystyczne kary na Wykonawcę.",
    "Z jednej strony organy podatkowe wydają korzystne interpretacje, z drugiej strony sądy administracyjne orzekają inaczej.",
    "Z jednej strony regulamin chroni prywatność, z drugiej dba o bezpieczeństwo."
]

antithesis = [
    "Nie chodzi tu o formę prawną, lecz o ostateczny skutek.",
    "Nie jest to kwestia interpretacji przepisów, lecz samego faktu naruszenia.",
    "Nie chodzi o literalne brzmienie umowy, lecz o zgodny zamiar stron.",
    "Nie mówimy tu o drobnych uchybieniach, lecz o rażącym niedbalstwie."
]

concessive = [
    "Choć ustawa przewiduje zwolnienie, to jednak organy dokonują interpretacji zawężającej.",
    "Mimo że umowa określa termin, to jednak w praktyce bywa on przedłużany.",
    "Choć powód przedstawił dowody, to jednak Sąd pierwszej instancji je pominął.",
    "Nawet jeśli regulamin dopuszcza taką możliwość, to jednak wiąże się to z ryzykiem."
]

connectors = ["Ponadto, ", "Co więcej, ", "Z kolei, ", "Natomiast, ", "Oraz "]

categories = {
    "umowa": {
        "core": ["Wykonawca zobowiązany jest do usunięcia wad", "Zamawiający zastrzega sobie prawo do odstąpienia", "Strony zgodnie postanawiają poddać spór pod arbitraż"],
        "end": ["Wszelkie zmiany wymagają formy pisemnej.", "Umowa wchodzi w życie z dniem podpisania."]
    },
    "pismo_procesowe": {
        "core": ["Sąd pierwszej instancji dokonał błędnej oceny", "Pozwany wniósł o oddalenie powództwa w całości", "Powód domaga się zasądzenia kwoty głównej"],
        "end": ["Mając powyższe na uwadze, wnoszę jak na wstępie.", "Zarzuty apelacji są w pełni uzasadnione."]
    },
    "regulamin": {
        "core": ["Administrator przetwarza dane osobowe w trybie ciągłym", "Użytkownik zobowiązany jest do zachowania poufności", "Usługodawca nie ponosi odpowiedzialności za szkody"],
        "end": ["Regulamin podlega prawu polskiemu.", "Konto użytkownika może zostać usunięte bez ostrzeżenia."]
    },
    "opinia_prawna": {
        "core": ["w odpowiedzi na zapytanie wskazujemy, iż ryzyko jest wysokie", "zgodnie z ugruntowaną linią orzeczniczą SN", "rekomendujemy podjęcie następujących działań naprawczych"],
        "end": ["Powyższe wnioski opierają się na aktualnym stanie prawnym.", "Dalsze kroki zależą od decyzji zarządu spółki."]
    }
}

random.seed(42)

for cat_name, cat_data in categories.items():
    for i in range(10):  # 10 documents per category = 40 total
        doc_sentences = []
        
        s1 = random.choice(openings) + random.choice(nominalizations) + ", " + random.choice(cat_data["core"]) + "."
        doc_sentences.append(s1)
        
        s2 = "Kwestia ta " + random.choice(empty_emphasis) + ", a " + random.choice(["fakt", "przepis", "ustawa", "dokument"]) + random.choice(typographical) + "potwierdza tę tezę."
        doc_sentences.append(s2)
        
        s3 = random.choice([random.choice(balanced_pairs), random.choice(antithesis), random.choice(concessive)])
        doc_sentences.append(s3)
        
        s4 = random.choice(connectors) + random.choice(cat_data["end"])
        doc_sentences.append(s4)
        
        text = " ".join(doc_sentences)
        
        # Capitalize first letters properly
        text = text[0].upper() + text[1:]
        
        filename = f"paper_evaluation/dataset/{cat_name}_{i+1:02d}.txt"
        with open(filename, "w", encoding="utf-8") as f:
            f.write(text)

print("Wygenerowano 40 dokumentów testowych w 'paper_evaluation/dataset/'")
