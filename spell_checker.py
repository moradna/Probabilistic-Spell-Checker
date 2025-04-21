import re
from collections import Counter
import math
import random
from nltk import ngrams
from nltk.tokenize import sent_tokenize
import nltk


# nltk.download('punkt')


class Spell_Checker:
    """The class implements a context sensitive spell checker. The corrections
        are done in the Noisy Channel framework, based on a language model and
        an error distribution model.
    """

    def __init__(self, lm=None):
        """Initializing a spell checker object with a language model as an
        instance  variable.

        Args:
            lm: a language model object. Defaults to None.
        """
        self.lm = lm
        self.error_tables = None
        self.text = None
        self.candidates_1 = {}
        self.candidates_2 = {}

    def add_language_model(self, lm):
        """Adds the specified language model as an instance variable.
            (Replaces an older LM dictionary if set)

            Args:
                lm: a Spell_Checker.Language_Model object
        """
        self.lm = lm

    def add_error_tables(self, error_tables):
        """ Adds the specified dictionary of error tables as an instance variable.
            (Replaces an older value dictionary if set)

            Args:
            error_tables (dict): a dictionary of error tables in the format
            of the provided confusion matrices:
            https://www.dropbox.com/s/ic40soda29emt4a/spelling_confusion_matrices.py?dl=0
        """
        self.error_tables = error_tables

    def evaluate_text(self, text):
        """Returns the log-likelihood of the specified text given the language
            model in use. Smoothing should be applied on texts containing OOV words
    
           Args:
               text (str): Text to evaluate.
    
           Returns:
               Float. The float should reflect the (log) probability.
        """
        if self.lm is None:
            return 0.0
        return self.lm.evaluate(text)

    def spell_check(self, text, alpha):
        """ Returns the most probable fix for the specified text. Use a simple
            noisy channel model if the number of tokens in the specified text is
            smaller than the length (n) of the language model.

            Args:
                text (str): the text to spell check.
                alpha (float): the probability of keeping a lexical word as is.

            Return:
                A modified string (or a copy of the original if no corrections are made.)
        """
        if self.lm is None or self.error_tables is None:
            return text
        sentences = sent_tokenize(text)
        text_fix = []
        sentences = [normalize_text(line) for line in sentences]
        for line in sentences:
            if not self.lm.chars:
                line = line.split()
            else:
                line = [char for char in line]

            word_error = self.wrong_word(line)  # first i check if have any error in the line

            if word_error is False:
                range_ind = (
                    self.lm.get_model_window_size() - 1,
                    len(line))  # in this case i looking for context wrong in the line
            else:
                range_ind = (word_error[1], word_error[1] + 1)  # focus on the wrong word

            for i in range(range_ind[0], range_ind[1]):
                candidates = self.candidates(line[i])
                if candidates == []:
                    continue

                prior_words = line[0:i]
                max_p = 0.0

                # Calculate the probability of each candidate word:
                for candidate in candidates:

                    word_candidate = candidate[0]
                    operations_candidate = candidate[1:]
                    if self.lm.chars:
                        sentence_candidate = "".join(prior_words + [word_candidate])
                    else:
                        sentence_candidate = " ".join(prior_words + [word_candidate])  # n_gram for evaluate text

                    P_ngram = math.pow(math.e, self.lm.evaluate_text(sentence_candidate))  # score ngram

                    P_x_w = 1  # P(x|w)
                    if operations_candidate is not None:
                        for j in range(0, len(operations_candidate)):
                            if operations_candidate[j] is not None:
                                type_error = operations_candidate[j][0]
                                value_error = operations_candidate[j][1]
                                if value_error not in self.error_tables[type_error]:
                                    continue
                                error_score = 0.0
                                for e in self.error_tables.keys():
                                    if value_error in self.error_tables[e]:
                                        error_score += self.error_tables[e][value_error]
                                if error_score == 0.0:
                                    error_score = 1
                                P_x_w *= (self.error_tables[type_error][value_error] / error_score) if word_candidate == \
                                                                                                       line[i] else (
                                                                                                                            1 - alpha) * P_x_w * P_ngram
                    candidate_p = alpha * P_ngram
                    if candidate_p >= max_p:
                        max_p = candidate_p
                        max_p_candidate = candidate

                line[i] = max_p_candidate[0]  # fix the line
            text_fix += (line)  # add to the new text
        if self.lm.chars:
            text_result = "".join(text_fix)
        else:
            text_result = " ".join(text_fix)
        return text_result

    def candidates(self, word):
        "Generate possible spelling corrections for word."
        c1 = list(self.edits1(word))
        c2 = list(self.edits2(word))
        c = set(c1 + c2 + [(word, None)])
        final = []
        for i in c:
            if i[0] in self.lm.vocab.keys():
                final.append(i)

        return final

    def edits1(self, word):
        """
        return candidates with edit distace 1.
        :param word: string
        :return: set(candidates)
        """
        letters = 'abcdefghijklmnopqrstuvwxyz'
        splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
        deletes = [(L + R[1:], ("deletion", L[len(L) - 1] + R[0] if len(L) > 0 else '#' + R[0])) for L, R in splits if
                   R]
        transposes = [(L + R[1] + R[0] + R[2:], ("transposition", R[0] + R[1])) for L, R in splits if len(R) > 1]
        replaces = [(L + c + R[1:], ("substitution", R[0] + c)) for L, R in splits if R for c in letters]
        inserts = [(L + c + R, ("insertion", L[len(L) - 1] + c if len(L) > 0 else '#' + c)) for L, R in splits for c in
                   letters]
        self.candidates_1[word] = set(deletes + transposes + replaces + inserts)
        return set(deletes + transposes + replaces + inserts)

    def edits2(self, word):
        """
        return candidates with edit distace 2.
        :param word: string
        :return: set(candidates)
        """
        merged = set()
        for c, op1 in self.candidates_1[word]:

            candidates_2 = self.edits1(c)
            for c2, op2 in candidates_2:
                if c2 != word:
                    merged.add((c2, op1, op2))

        merged = list(merged)
        self.candidates_2[word] = merged
        return merged

    def wrong_word(self, sentence):
        for i in range(len(sentence)):
            word = sentence[i]
            if word not in self.lm.vocab.keys():
                return (word, i)
        return False

    #####################################################################
    #                   Inner class                                     #
    #####################################################################

    class Language_Model:
        """The class implements a Markov Language Model that learns a model from a given text.
            It supports language generation and the evaluation of a given string.
            The class can be applied on both word level and character level.
        """

        def __init__(self, n=3, chars=False):
            """Initializing a language model object.
            Args:
                n (int): the length of the markov unit (the n of the n-gram). Defaults to 3.
                chars (bool): True iff the model consists of ngrams of characters rather than word tokens.
                              Defaults to False
            """
            self.n = n
            self.chars = chars
            self.dict_prefix_ngram = {}  # {N-1 gram:list(next_word)}
            self.vocab = {}  # {word:count} #change to vocab
            self.count_prefix_ngram = {}  # {N-1 gram:count}
            self.model_dict = None  # a dictionary of the form {ngram:count}, holding counts of all ngrams in the specified text.
            # NOTE: This dictionary format is inefficient and insufficient (why?), therefore  you can (even encouraged to)
            # use a better data structure.
            # However, you are requested to support this format for two reasons:
            # (1) It is very straight forward and force you to understand the logic behind LM, and
            # (2) It serves as the normal form for the LM so we can call get_model_dictionary() and peek into you model.

        def build_model(self, text):
            """populates the instance variable model_dict.

                Args:
                    text (str): the text to construct the model from.
            """
            self.model_dict = {}
            text = normalize_text(text)  # nornalize
            if not self.chars:
                text_split = text.split()
            else:
                text_split = [char for char in text]
            self.vocab = self.build_word_vocab(text_split) if not self.chars else self.build_word_vocab(
                text.split())  # build word vocavbory
            for i in range(len(text_split) - self.n + 1):
                n_gram = text_split[i:i + self.n]
                if self.chars:
                    n_gram_text = "".join(n_gram)
                else:
                    n_gram_text = " ".join(n_gram)
                if n_gram_text not in self.model_dict.keys():
                    self.model_dict[n_gram_text] = 1
                else:
                    self.model_dict[n_gram_text] += 1

            for i in range(len(text_split) - self.n + 1):
                n_gram = text_split[i:i + self.n - 1]
                if self.chars:
                    n_gram_text = "".join(n_gram)
                else:
                    n_gram_text = " ".join(n_gram)
                next_word = text_split[i + self.n - 1:i + self.n][0]
                if n_gram_text not in self.dict_prefix_ngram:
                    self.dict_prefix_ngram[n_gram_text] = [next_word]
                    self.count_prefix_ngram[n_gram_text] = 1
                else:
                    self.dict_prefix_ngram[n_gram_text].append(next_word)
                    self.count_prefix_ngram[n_gram_text] += 1

        def get_model_dictionary(self):
            """Returns the dictionary class object
            """
            return self.model_dict

        def get_model_window_size(self):
            """Returning the size of the context window (the n in "n-gram")
            """
            return self.n

        def generate(self, context=None, n=20):
            """Returns a string of the specified length, generated by applying the language model
            to the specified seed context. If no context is specified the context should be sampled
            from the models' contexts distribution. Generation should stop before the n'th word if the
            contexts are exhausted. If the length of the specified context exceeds (or equal to)
            the specified n, the method should return a prefix of length n of the specified context.

                Args:
                    context (str): a seed context to start the generated string from. Defaults to None
                    n (int): the length of the string to be generated.

                Return:
                    String. The generated text.

            """
            if self.model_dict is None or n == 0:
                return context
            if context is None:
                context = self.random_context()
                context_split = context.split() if not self.chars else list(context)
            else:
                context_split = context.split() if not self.chars else list(context)

            if len(context_split) >= n:
                if self.chars:
                    return "".join(context_split[:n])
                else:
                    return " ".join(context_split[:n])

            generate_line = context_split
            while n > len(generate_line):
                if context not in self.dict_prefix_ngram:
                    context = self.random_context()
                word_weghit = Counter(self.dict_prefix_ngram[context])
                next_word = random.choices(list(word_weghit.keys()), weights=list(word_weghit.values()))[0]
                generate_line.append(next_word)
                if self.chars:
                    context = "".join(generate_line[:-(self.n - 1)])
                else:
                    context = " ".join(generate_line[:-(self.n - 1)])

            if self.chars:
                result = "".join(generate_line)
            else:
                result = " ".join(generate_line)
            return result

        def random_context(self):
            weights, context_keys = [], []
            for key, value in self.dict_prefix_ngram.items():
                context_keys.append(key)
                weights.append(len(value))
            return random.choices(context_keys, weights, k=1)[0]

        def evaluate_text(self, text):
            """Returns the log-likelihood of the specified text to be a product of the model.
               Laplace smoothing should be applied if necessary.

               Args:
                   text (str): Text to evaluate.

               Returns:
                   Float. The float should reflect the (log) probability.
            """
            if text is None:
                return 0.0
            if not self.chars:
                split_text = text.split()
            else:
                split_text = [char for char in text]
            if len(split_text) < self.n:
                return 0.0
            prob = 1
            for i in range(len(split_text) - self.n + 1):
                curr_gram = ' '.join(split_text[i:i + self.n])
                n_gram_curr = ' '.join(split_text[i:i + self.n - 1])
                if curr_gram in self.model_dict and n_gram_curr in self.dict_prefix_ngram:
                    prob *= self.model_dict[curr_gram] / self.count_prefix_ngram[n_gram_curr]
                else:
                    prob *= self.smooth(curr_gram)
            return math.log(prob, 10)

        def smooth(self, ngram):
            """Returns the smoothed (Laplace) probability of the specified ngram.

                Args:
                    ngram (str): the ngram to have its probability smoothed

                Returns:
                    float. The smoothed probability.
            """
            if not self.chars:
                ngram = ngram.split()
            else:
                ngram = [char for char in ngram]

            context = ngram[:-1]
            next_word = ngram[-1]

            full_gram = ' '.join(context)
            context_n_gran = ' '.join(next_word)

            if full_gram in self.model_dict.keys():
                up = self.model_dict[full_gram] + 1
            else:
                up = 1
            if context_n_gran in self.count_prefix_ngram.keys():

                down = self.count_prefix_ngram[context_n_gran] + len(self.vocab)
            else:
                down = len(self.vocab)
            return up / down

        def build_word_vocab(self, text):
            """
            Build word vocabulary
            input:
                text (list): list (tokens)
            output:
                    (dict): {word: count}
            """
            word_dict = {}
            for word in text:
                if word in word_dict:
                    word_dict[word] += 1
                else:
                    word_dict[word] = 1
            return word_dict


def normalize_text(text):
    """Returns a normalized version of the specified string.
      You can add default parameters as you like (they should have default values!)
      You should explain your decisions in the header of the function.

      Args:
        text (str): the text to normalize

      Returns:
        string. the normalized text.
    """
    text = text.lower()
    separated_text = ' '.join(text.split('<s>'))
    lowered_text = separated_text.lower()
    set_punctuations = {char for char in '''!()-[]{};:'"\,<>./?@#$%^&*_~'''}
    set_punctuations.add('\n')
    normalized_text = ''
    for index in range(len(lowered_text)):
        try:
            if lowered_text[index] not in set_punctuations:
                normalized_text += lowered_text[index]
            else:
                try:
                    if lowered_text[index + 1] != ' ' and lowered_text[index - 1] != ' ':
                        normalized_text += ' '
                except IndexError:
                    continue
        except UnicodeDecodeError:
            continue
    return normalized_text


def who_am_i():  # this is not a class method
    """Returns a ductionary with your name, id number and email. keys=['name', 'id','email']
        Make sure you return your own info!
    """
    return {'name': 'Natalie Morad', 'id': '000', 'email': 'moradna@post.bgu.ac.il'}
