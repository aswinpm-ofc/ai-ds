from collections import Counter, defaultdict

class NGramModel:
    def __init__(self, tokens):
        self.unigrams = Counter(tokens)
        self.bigrams = defaultdict(Counter)
        self.trigrams = defaultdict(Counter)

        for i in range(len(tokens) - 1):
            self.bigrams[tokens[i]][tokens[i + 1]] += 1

        for i in range(len(tokens) - 2):
            key = (tokens[i], tokens[i + 1])
            self.trigrams[key][tokens[i + 2]] += 1

    def next_words(self, context):
        if len(context) >= 2:
            key = tuple(context[-2:])
            if key in self.trigrams:
                return self.trigrams[key]

        if context:
            word = context[-1]
            if word in self.bigrams:
                return self.bigrams[word]

        return self.unigrams
