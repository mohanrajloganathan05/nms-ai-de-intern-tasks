from nltk.corpus import wordnet


class DictionaryHelper:

    def get_meaning_and_synonyms(self, word):

        meanings = []
        synonyms = []

        synsets = wordnet.synsets(word)

        # If word not found
        if len(synsets) == 0:

            return {
                "meanings": [],
                "synonyms": []
            }

        # Get meanings
        for synset in synsets:

            meaning = synset.definition()

            if meaning not in meanings:
                meanings.append(meaning)

            # Get synonyms
            for lemma in synset.lemmas():

                synonym = lemma.name()

                if synonym not in synonyms:
                    synonyms.append(synonym)

        return {
            "meanings": meanings[:3],
            "synonyms": synonyms[:5]
        }