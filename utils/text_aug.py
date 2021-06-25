# from textaugment import Wordnet
# from textaugment import EDA
import random

import nlpaug.flow as naf
import nlpaug.augmenter.char as nac
import nlpaug.augmenter.word as naw

def augment_text(text):
    aug = naf.Sometimes([
        nac.RandomCharAug(action="delete"),
        nac.RandomCharAug(action="insert"),
        naw.RandomWordAug(action="swap"),
        naw.SynonymAug()
    ])
    return aug.augment(text)


# def text_augment(text):
    
#     aug_text = text
#     aug_mode = random.randint(0, 5)
   
#     if aug_mode == 0:
#         # back_translation_aug = naw.BackTranslationAug()
#         # aug_text = back_translation_aug.augment(text)
#         aug = naw.RandomWordAug(action="swap")
        
#         aug_text = aug.augment(text)
#     elif aug_mode == 1:

#     # if mode == 0:
#     #     aug_text = wordnet.augment(text)
#     # elif mode == 1:
#     #     # original deletion, but have bugs
#     #     aug_text = eda.synonym_replacement(text)
#     # elif mode == 2:
#     #     aug_text = eda.random_swap(text)
#     # elif mode == 3:
#     #     aug_text = eda.synonym_replacement(text)
#     # elif mode == 4:
#     #     # original insertion , maybe bugs???
#     #     aug_text = eda.random_swap(text)
#     # else:
#     #     aug_text = text

#     return aug_text


