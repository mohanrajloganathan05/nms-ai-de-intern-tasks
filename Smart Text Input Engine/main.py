from modules.predictor import NextWordPredictor
from modules.spell_corrector import SpellCorrector
from modules.grammar_corrector import GrammarCorrector
from modules.dictionary_helper import DictionaryHelper


print("====================================")
print("   SMART TEXT INPUT ENGINE")
print("====================================")


# Create objects
predictor = NextWordPredictor()
spell = SpellCorrector()
grammar = GrammarCorrector()
dictionary = DictionaryHelper()


print("\nTraining AI model...")
predictor.train_model()
print("Training completed!")


while True:

    text = input("\nEnter Text: ")

    # Exit condition
    if text.lower() == "exit":
        print("\nProgram Closed")
        break

    # Detect unknown words
    unknown_words = spell.detect_unknown_words(text)

    # Spell correction
    spell_corrected = spell.correct_spelling(text)

    # Grammar correction
    grammar_corrected = grammar.correct_grammar(spell_corrected)

    # Next-word prediction
    suggestions = predictor.predict_next_word(grammar_corrected)

    # Dictionary support
    words = grammar_corrected.split()

    if len(words) > 0:
        last_word = words[-1]
    else:
        last_word = ""

    dictionary_data = dictionary.get_meaning_and_synonyms(last_word)

    print("\n------------------------------------")

    # Original text
    print("\nOriginal Text:")
    print(text)

    # Unknown words
    print("\nUnknown Words:")

    if len(unknown_words) == 0:
        print("No unknown words")

    else:
        for word in unknown_words:
            print("•", word)

    # Spell corrected text
    print("\nSpell Corrected:")
    print(spell_corrected)

    # Grammar corrected text
    print("\nGrammar Corrected:")
    print(grammar_corrected)

    # Dictionary meanings
    print("\nDictionary Support:")

    print("\nMeanings:")

    if len(dictionary_data["meanings"]) == 0:
        print("No meanings found")

    else:
        for meaning in dictionary_data["meanings"]:
            print("•", meaning)

    # Synonyms
    print("\nSynonyms:")

    if len(dictionary_data["synonyms"]) == 0:
        print("No synonyms found")

    else:
        for synonym in dictionary_data["synonyms"]:
            print("•", synonym)

    # Next-word suggestions
    print("\nNext Word Suggestions:")

    if len(suggestions) == 0:
        print("No suggestions found")

    else:
        for word in suggestions:
            print("•", word)

    print("\n------------------------------------")