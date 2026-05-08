import language_tool_python


class GrammarCorrector:

    def __init__(self):

        self.tool = language_tool_python.LanguageTool('en-US')

    def correct_grammar(self, text):

        matches = self.tool.check(text)

        corrected_text = language_tool_python.utils.correct(text, matches)

        return corrected_text