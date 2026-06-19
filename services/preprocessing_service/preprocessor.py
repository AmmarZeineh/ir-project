import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize
from typing import Optional

STOP_WORDS = set(stopwords.words("english"))
stemmer     = PorterStemmer()
lemmatizer  = WordNetLemmatizer()


def normalize(text: str) -> str:
    """lowercase + إزالة punctuation + أرقام + مسافات زيادة"""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text) 
    text = re.sub(r"\d+", " ", text)         
    text = re.sub(r"\s+", " ", text).strip() 
    return text


def tokenize(text: str) -> list[str]:
    return word_tokenize(text)


def remove_stopwords(tokens: list[str]) -> list[str]:
    return [t for t in tokens if t not in STOP_WORDS and len(t) > 1]


def stem(tokens: list[str]) -> list[str]:
    return [stemmer.stem(t) for t in tokens]


def lemmatize(tokens: list[str]) -> list[str]:
    return [lemmatizer.lemmatize(t) for t in tokens]


def preprocess(
    text: str,
    use_stemming: bool = False,
    use_lemmatization: bool = True,
    remove_stops: bool = True,
) -> dict:
    """
    Pipeline كامل — بيرجع dict فيه كل المراحل
    """
    normalized = normalize(text)
    tokens     = tokenize(normalized)

    if remove_stops:
        tokens = remove_stopwords(tokens)

    stemmed    = stem(tokens)    if use_stemming      else []
    lemmatized = lemmatize(tokens) if use_lemmatization else []

    if use_lemmatization:
        final = lemmatized
    elif use_stemming:
        final = stemmed
    else:
        final = tokens

    return {
        "original":   text,
        "normalized": normalized,
        "tokens":     tokens,
        "stemmed":    stemmed,
        "lemmatized": lemmatized,
        "final":      final,                    
        "final_str":  " ".join(final),          
    }