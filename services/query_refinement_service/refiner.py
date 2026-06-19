from spellchecker import SpellChecker
from nltk.corpus import wordnet
from collections import Counter
import pandas as pd
import pickle
from pathlib import Path

from spellchecker import SpellChecker

spell = SpellChecker()

CUSTOM_WORDS = {
    "wifi", "bluetooth", "ww1", "ww2", "covid", "ai", "ml",
    "api", "gpu", "cpu", "bsn", "aml", "rsa", "cdg", "lps",
    "sigmet", "legionella", "pneumophila", "dysarthria",
    "ventilation", "medicares", "midsegment", "trapezoid",
    "sous", "vide", "determinants", "contour", "plowing",
    "salvation", "visceral", "tracheids", "theraderm", "famvir",
    "goldfish", "koi", "axon", "synaptic",
}
spell.word_frequency.load_words(CUSTOM_WORDS)

HISTORY_PATH = Path("data/processed/search_history.pkl")

def _load_history() -> list[str]:
    if HISTORY_PATH.exists():
        with open(HISTORY_PATH, "rb") as f:
            return pickle.load(f)
    return []

def _save_history(history: list[str]):
    with open(HISTORY_PATH, "wb") as f:
        pickle.dump(history, f)

def add_to_history(query: str):
    history = _load_history()
    history.append(query.lower().strip())
    _save_history(history)

def spell_correct(query: str) -> str:
    words     = query.lower().split()
    corrected = [spell.correction(w) or w for w in words]
    return " ".join(corrected)


def expand_with_synonyms(query: str, max_synonyms: int = 2) -> str:
    words    = query.lower().split()
    expanded = list(words)

    for word in words:
        syns = set()
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                name = lemma.name().replace("_", " ")
                if name != word:
                    syns.add(name)
            if len(syns) >= max_synonyms:
                break
        expanded.extend(list(syns)[:max_synonyms])

    return " ".join(expanded)


def suggest_from_history(partial: str, top_n: int = 5) -> list[str]:
    history = _load_history()
    partial = partial.lower().strip()
    matches = [q for q in history if q.startswith(partial) and q != partial]
    counter = Counter(matches)
    return [q for q, _ in counter.most_common(top_n)]

def refine_query(query: str, use_synonyms: bool = True) -> dict:
    original  = query
    corrected = spell_correct(query)
    expanded  = expand_with_synonyms(corrected) if use_synonyms else corrected
    suggestions = suggest_from_history(query)

    add_to_history(original)

    return {
        "original":    original,
        "corrected":   corrected,
        "expanded":    expanded,
        "suggestions": suggestions,
        "changed":     original.lower() != corrected.lower(),
    }