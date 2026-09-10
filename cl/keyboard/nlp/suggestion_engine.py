import re
from .preprocess import clean_text, tokenize
from .ngram_model import NGramModel

class SuggestionEngine:
    def __init__(self, corpus_path):
        with open(corpus_path, "r", encoding="utf-8") as f:
            raw = f.read()

        cleaned = clean_text(raw)
        tokens = tokenize(cleaned)
        self.model = NGramModel(tokens)

        # Vocabulary sorted by corpus frequency.
        self.vocabulary = set(self.model.unigrams.keys())

    def suggest(self, text, limit=5):
        text = text.lower()

        # Regex detects the incomplete word at the end of the input.
        match = re.search(r"([a-z]+)$", text)
        prefix = match.group(1) if match else ""

        # Text before the incomplete final word is used as context.
        context_text = text[:match.start()] if match else text
        context = tokenize(context_text)

        # Get context candidates from trigram/bigram model.
        contextual = self.model.next_words(context)

        # Prefix candidates from corpus vocabulary.
        prefix_candidates = {
            word for word in self.vocabulary
            if word.startswith(prefix)
        }

        if prefix:
            # Strongly favor words matching the typed prefix.
            candidates = prefix_candidates
        else:
            candidates = set(contextual.keys())

        scored = []
        for word in candidates:
            if prefix and not word.startswith(prefix):
                continue

            context_score = contextual.get(word, 0)
            frequency_score = self.model.unigrams.get(word, 0)

            # Context gets priority; corpus frequency breaks ties.
            score = (context_score * 10) + frequency_score
            scored.append((score, word))

        scored.sort(key=lambda x: (-x[0], x[1]))
        return [word for _, word in scored[:limit]]
