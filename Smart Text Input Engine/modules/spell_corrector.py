from textblob import TextBlob
from nltk.corpus import brown


class SpellCorrector:

    def __init__(self):

        # Create vocabulary set
        self.vocabulary = set()

        words = brown.words()

        for word in words:
            self.vocabulary.add(word.lower())

    def correct_spelling(self, text):

        words = text.split()

        corrected_sentence = []

        for word in words:

            lower_word = word.lower()

            # If word exists in vocabulary
            if lower_word in self.vocabulary:

                corrected_sentence.append(word)

            else:

                # Try correction
                blob = TextBlob(word)

                corrected_word = str(blob.correct())

                corrected_sentence.append(corrected_word)

        final_text = " ".join(corrected_sentence)

        return final_text
    
    
    
    def detect_unknown_words(self, text):

        words = text.split()

        unknown_words = []

        for word in words:

            if word.lower() not in self.vocabulary:
                unknown_words.append(word)

        return unknown_words