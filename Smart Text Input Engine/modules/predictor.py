from nltk.corpus import brown


class NextWordPredictor:

    def __init__(self):

        # Store word relationships
        self.model = {}

    def train_model(self):

        # Get all words from dataset
        words = brown.words()

        # Convert words to lowercase
        lower_words = []

        for word in words:
            lower_words.append(word.lower())

        # Build trigram model
        for i in range(len(lower_words) - 2):

            word1 = lower_words[i]
            word2 = lower_words[i + 1]
            next_word = lower_words[i + 2]

            # Create tuple key
            key = (word1, word2)

            # If key not present
            if key not in self.model:
                self.model[key] = []

            # Add next word
            self.model[key].append(next_word)

    def predict_next_word(self, text):

        # Convert input to lowercase
        text = text.lower()

        # Split sentence into words
        words = text.split()

        # Need minimum 2 words
        if len(words) < 2:
            return []

        # Last 2 words
        word1 = words[-2]
        word2 = words[-1]

        key = (word1, word2)

        # If not found
        if key not in self.model:
            return []

        # Get all next words
        suggestions = self.model[key]

        # Remove duplicates manually
        unique_words = []

        for word in suggestions:

            if word not in unique_words:
                unique_words.append(word)

        # Return first 5 words
        return unique_words