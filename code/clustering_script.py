#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cluster deutsche Chatbot-Prompts in Usecases mit BERTopic.

- Liest eine JSON-Datei mit einem Array von Strings (Prompts).
- Filtert einfache Chitchat-Prompts ("hallo", "danke", "was kannst du") und
  markiert mögliche Abuse-Prompts (über einfache Wortliste).
- Erzeugt Satz-Embeddings mit:
    sentence-transformers/paraphrase-multilingual-mpnet-base-v2
- Führt Topic-Clustering mit BERTopic durch.
- Schreibt eine CSV mit allen gefilterten (inhaltlich relevanten) Prompts
  und den zugehörigen Topic-IDs (Clusters) und Topic-Labels (Top-Wörter).

Beispielaufruf:
    python cluster_prompts.py \
        --input prompts.json \
        --output clustered_prompts.csv \
        --min-topic-size 20

Benötigte Pakete (vorher installieren):
    pip install pandas sentence-transformers bertopic umap-learn hdbscan
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from bertopic import BERTopic


# ----------------------------- Konfiguration ----------------------------- #

# Regexe für reine Meta/Chitchat-Prompts (genauer Match auf ganze Eingabe)
META_REGEXES = [
    re.compile(r'^(hallo|hi|hey|hallo zusammen|moin|servus|grüß dich|'
               r'guten (morgen|tag|abend))!?$', re.IGNORECASE),
    re.compile(r'^(danke|dankeschön|vielen dank|thx|merci)!?$', re.IGNORECASE),
    re.compile(r'^test!?$', re.IGNORECASE),
    re.compile(r'^was kannst du( so| alles)?\??$', re.IGNORECASE),
    re.compile(r'^bist du (echt|ein bot)\??$', re.IGNORECASE),
]

# Einfache Abuse-/Beleidigungs-Wortliste (bitte projektspezifisch ergänzen)
# -> hier KEINE expliziten slurs, nur neutrale Platzhalter/Wörter.
ABUSE_WORDS = {
    # Beispiele für allgemeine Beschimpfungen:
    "idiot",
    "depp",
    "arsch",
    "blödmann",
    "dummkopf",
    "fotze",
    "hure",
    "neger"
    # Füge hier projektspezifische Ausdrücke hinzu, falls gewünscht.
}

# Embedding-Modell (empfohlen für Deutsch)
EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"


# ----------------------------- Hilfsfunktionen --------------------------- #

def load_sentences_from_json(path: Path) -> pd.DataFrame:
    """Lädt Sätze aus einer JSON-Datei (Array von Strings oder Dict mit 'sentences')."""
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        sentences = data
    elif isinstance(data, dict) and "sentences" in data:
        sentences = data["sentences"]
    else:
        raise ValueError(
            "JSON muss entweder ein Array von Strings oder ein Dict mit dem Key 'sentences' sein."
        )

    # Nur Strings behalten, Rest verwerfen
    sentences = [str(s) for s in sentences if isinstance(s, (str, int, float))]
    df = pd.DataFrame({"text": sentences})
    return df


def is_meta_chitchat(text: str) -> bool:
    """True, wenn der Prompt ein reiner Gruß/Danke/Test/Meta-Prompt ist."""
    t = text.strip()
    for pattern in META_REGEXES:
        if pattern.match(t):
            return True
    return False


def contains_abuse(text: str) -> bool:
    """True, wenn der Prompt offensichtliche Beschimpfungen enthält."""
    # Einfache Tokenisierung über Wortzeichen
    tokens = re.findall(r"\w+", text.lower())
    return any(tok in ABUSE_WORDS for tok in tokens)


def classify_rule_based(text: str) -> str:
    """
    Eingaben grob klassifizieren:
      - 'meta_chitchat' : reine Hallo/Danke/Test/Was-kannst-du-Prompts
      - 'abuse'         : offensichtliche Beleidigungen
      - 'empty'         : leer oder nur Whitespace
      - 'content_candidate' : alles andere (wird geclustert)
    """
    if text is None:
        return "empty"
    t = str(text).strip()
    if not t:
        return "empty"

    if contains_abuse(t):
        return "abuse"

    if is_meta_chitchat(t):
        return "meta_chitchat"

    return "content_candidate"


def build_embedding_model(device: str = None) -> SentenceTransformer:
    """Lädt das SentenceTransformer-Modell auf CPU oder GPU."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Lade Embedding-Modell '{EMBEDDING_MODEL_NAME}' auf Gerät: {device}")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME, device=device)
    return model


def run_bertopic(
    docs_unique: list[str],
    embedding_model: SentenceTransformer,
    min_topic_size: int = 30,
) -> tuple[BERTopic, np.ndarray, np.ndarray]:
    """
    Führt BERTopic-Clustering auf den eindeutigen Prompts aus.

    Gibt zurück:
        topic_model : trainiertes BERTopic-Modell
        topics      : Topic-ID für jedes Dokument (gleiche Länge wie docs_unique)
        probs       : Topic-Wahrscheinlichkeiten (Array shape: (n_docs, n_topics))
    """
    print(f"[INFO] Starte BERTopic mit min_topic_size={min_topic_size} für {len(docs_unique)} einzigartige Prompts.")

    topic_model = BERTopic(
        embedding_model=embedding_model,
        language="german",
        min_topic_size=min_topic_size,
        calculate_probabilities=True,
        verbose=True,
    )

    topics, probs = topic_model.fit_transform(docs_unique)

    return topic_model, np.array(topics), probs


def build_topic_representations(topic_model: BERTopic) -> dict[int, str]:
    """
    Erzeugt für jedes Topic eine kurze Text-Representation (Top-Wörter).
    Returns dict: topic_id -> "wort1, wort2, wort3, ..."
    """
    topics_info = topic_model.get_topic_info()
    topic_reprs: dict[int, str] = {}

    for topic_id in topics_info["Topic"].unique():
        if topic_id == -1:
            topic_reprs[topic_id] = "Outlier"
            continue

        words_scores = topic_model.get_topic(topic_id)  # Liste von (Wort, Score)
        if not words_scores:
            topic_reprs[topic_id] = f"Topic {topic_id}"
            continue

        top_words = [w for w, _ in words_scores[:5]]
        topic_reprs[topic_id] = ", ".join(top_words)

    return topic_reprs


# ----------------------------- Haupt-Workflow ---------------------------- #

def main():
    parser = argparse.ArgumentParser(description="Cluster deutsche Chatbot-Prompts mit BERTopic.")
    parser.add_argument(
        "--input", "-i", type=str, required=True,
        help="Pfad zur JSON-Datei mit einem Array von Prompts."
    )
    parser.add_argument(
        "--output", "-o", type=str, required=True,
        help="Pfad zur Ausgabe-CSV mit gefilterten Prompts und Cluster-Zuordnung."
    )
    parser.add_argument(
        "--min-topic-size", type=int, default=30,
        help="Minimalgröße eines Topics für BERTopic (Standard: 30)."
    )

    args = parser.parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    # 1) Daten laden
    print(f"[INFO] Lade Prompts aus: {input_path}")
    df = load_sentences_from_json(input_path)
    print(f"[INFO] Gesamtanzahl Prompts: {len(df)}")

    # 2) Regelbasierte Klassifikation (Meta/Abuse/Empty/Content)
    print("[INFO] Führe regelbasierte Klassifikation durch ...")
    df["category_rule_based"] = df["text"].apply(classify_rule_based)

    # Übersicht
    print("[INFO] Kategorien-Verteilung (rule-based):")
    print(df["category_rule_based"].value_counts())

    # 3) Nur inhaltlich relevante Kandidaten für Clustering behalten
    df_candidates = df[df["category_rule_based"] == "content_candidate"].copy()
    df_candidates["text"] = df_candidates["text"].astype(str).str.strip()

    # Optional: sehr kurze Texte entfernen (falls gewünscht)
    df_candidates = df_candidates[df_candidates["text"].str.len() >= 10]

    print(f"[INFO] Anzahl inhaltlich relevanter Kandidaten: {len(df_candidates)}")

    if len(df_candidates) == 0:
        print("[WARN] Keine content_candidate-Prompts gefunden. Beende.")
        return

    # 4) Duplikate für das Clustering entfernen (aber später wieder auf alle erweitern)
    df_candidates["text_key"] = df_candidates["text"].str.strip()
    df_unique = df_candidates.drop_duplicates(subset=["text_key"]).copy()
    df_unique.reset_index(drop=False, inplace=True)  # original Index behalten falls nötig

    print(f"[INFO] Einzigartige Prompts für Clustering: {len(df_unique)}")

    docs_unique = df_unique["text"].tolist()

    # 5) Embedding-Modell laden
    embedding_model = build_embedding_model()

    # 6) BERTopic-Clustering ausführen
    topic_model, topics_unique, probs_unique = run_bertopic(
        docs_unique=docs_unique,
        embedding_model=embedding_model,
        min_topic_size=args.min_topic_size,
    )

    # 7) Ergebnisse dem df_unique zuordnen
    df_unique["topic_id"] = topics_unique
    if probs_unique is not None:
        # höchste Wahrscheinlichkeit pro Dokument
        max_probs = probs_unique.max(axis=1)
        df_unique["topic_probability"] = max_probs
    else:
        df_unique["topic_probability"] = np.nan

    # 8) Topic-Representation (Top-Wörter) aufbauen
    topic_reprs = build_topic_representations(topic_model)
    df_unique["topic_repr"] = df_unique["topic_id"].map(topic_reprs)

    # 9) Topic-Zuordnung wieder auf alle Kandidaten ausrollen
    topic_map = dict(zip(df_unique["text_key"], df_unique["topic_id"]))
    prob_map = dict(zip(df_unique["text_key"], df_unique["topic_probability"]))
    repr_map = dict(zip(df_unique["topic_key"] if "topic_key" in df_unique.columns else df_unique["text_key"],
                        df_unique["topic_repr"]))

    df_candidates["topic_id"] = df_candidates["text_key"].map(topic_map)
    df_candidates["topic_probability"] = df_candidates["text_key"].map(prob_map)
    df_candidates["topic_repr"] = df_candidates["text_key"].map(
        {row["text_key"]: row["topic_repr"] for _, row in df_unique.iterrows()}
    )

    # 10) Finale Ausgabe vorbereiten
    # Nur gefilterte, inhaltlich relevante Sätze + Clusterinfos
    df_out = df_candidates.copy()
    df_out = df_out.rename(
        columns={
            "text": "sentence",
            "topic_id": "cluster_id",
            "topic_repr": "cluster_label",
            "topic_probability": "cluster_probability",
        }
    )

    # Optional: Spaltenreihenfolge
    df_out = df_out[
        [
            "sentence",
            "cluster_id",
            "cluster_label",
            "cluster_probability",
            "category_rule_based",
        ]
    ]

    # 11) CSV schreiben
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(output_path, index=False, encoding="utf-8")
    print(f"[INFO] Fertig. CSV mit {len(df_out)} gefilterten Sätzen geschrieben nach: {output_path}")


if __name__ == "__main__":
    main()