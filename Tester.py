from ex1 import *
from spelling_confusion_matrices import error_tables
from spelling_confusion_matrices import *

# import nltk
# nltk.download('punkt')
#

b=open("big.txt")
data=b.read()

normalized_corpus = normalize_text(data)
# print(normalized_corpus)


n = 0 # Set the n-gram size you want to use
lm = Spell_Checker.Language_Model(n=2, chars=False)   # Set chars=True for character-level language model
lm.build_model(normalized_corpus)
# print(lm.dict_prefix_ngram)

print(lm.generate(context="gutenberg book of",n=4))

spell_checker = Spell_Checker()
spell_checker.add_language_model(lm)


spell_checker.add_error_tables(error_tables)
# input_text = "the project gutenberg book"
# alpha = 0.0 # Adjust the alpha value based on your preference (higher alpha values give more weight to the original word)
# print("==================================")
# print(spell_checker)
# corrected_text = spell_checker.spell_check(input_text, alpha)
# print("Original text:", input_text)
# print("Corrected text:", corrected_text)
#
# print("==================================")
# input_text = "This is an exmple sentece with erors"
# alpha = 0.95 # Adjust the alpha value based on your preference (higher alpha values give more weight to the original word)
#
# corrected_text = spell_checker.spell_check(input_text, alpha)
# print("Original text:", input_text)
# print("Corrected text:", corrected_text)
#
# print("==================================")
#
# input_text = "of of the project gutenberg book of of of of of of of. This is an exmple sentece with erors"
# corrected_text = spell_checker.spell_check(input_text, alpha)
# print("Original text:", input_text)
# print("Corrected text:", corrected_text)
#
# print("==================================")
#
# #
# test_cases = [
#     ("beutiful", "beautiful"),
#     ("baautiful", "beautiful"),
#     ("bautiful", "beautiful"),
#     ("beutifull", "beautiful"),
#     ("beuatiful", "beautiful"),
#     ("inteligence", "intelligence"),
#     ("intellgence", "intelligence"),
#     ("intelligenc", "intelligence"),
#     ("intellignce", "intelligence"),
#     ("intelliegnce", "intelligence"),
#     ("intelilgence", "intelligence"),
#     ("recomend", "recommend"),
#     ("reccommend", "recommend"),
#     ("reccomend", "recommend"),
#     ("reecommand", "recommend"),
#     ("envirnoment", "environment"),
#     ("environmet", "environment"),
#     ("envronment", "environment"),
#     ("envirnomnt", "environment"),
#     ("enviroonmet", "environment"),
#     ("acess", "access"),
#     ("acount", "account"),
#     ("acomplish", "accomplish"),
#     ("adres", "address"),
#     ("adresable", "addressable"),
#     ("definate", "definite"),
#     ("definitly", "definitely"),
#     ("definitiv", "definitive"),
#     ("recive", "receive"),
#     ("recived", "received"),
#     ("recieving", "receiving"),
# # ]
# for i in test_cases:
#    a =  spell_checker.spell_check(i[0], alpha=0.95)
#    print(a ==i[1])

# print(lm.model_dict)