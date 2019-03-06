import os
import pandas as pd
import copy
from nltk.tokenize import word_tokenize
import ebm_nlp_demo as e
import re
from glob import glob
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import PyPDF2
from io import StringIO
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import numpy as np
from pycorenlp import StanfordCoreNLP
import pprint
import string
from nltk.stem.porter import PorterStemmer
from collections import Counter
import enchant
from tabulate import tabulate
import requests as req
from bs4 import BeautifulSoup
from numba import jit
import json
import ast

pd.options.display.max_rows =500000
Outcomes = {
     'No label':0,
     'Physical':1,
     'Pain':2,
     'Mortality': 3,
     'Adverse-effects':4,
     'Mental':5,
     'Other':6
}

class annotate_text:

    def __init__(self):
        self.stan = StanfordCoreNLP('http://localhost:9000')

        self.properties = {'annotators': 'pos',  'outputFormat': 'json'}
        stat_terms = []
        with open('stat_file.txt') as f:
            for i in f.readlines():
                stat_terms.append(i)

        self.unwanted = {'punctuation': ['.', "'", ';', ':'],
                    'stat_lexicon':  extracting_statistical_lexicon(scrap_stat_terms(scrap_source = [])),
                    'rct_instruments_results': ['subjective significance questionnaire',
                                                'Baseline tumour marker',
                                                'Stroop Test and Wisconsin Card Sorting Test',
                                                'visual analogue scale',
                                                'visual analog scale',
                                                'Stroop Test','Wisconsin Card Sorting Test','age', 'gender', '( VAS )', 'VAS',
                                                'efficacy and safety',
                                                'safety and efficacy',
                                                'effect',
                                                'area',
                                                'curve',
                                                '( )',
                                                'questionnaire',
                                                'Western Ontario and McMaster Universities Osteoarthritis Index ( WOMAC ) function scale',
                                                'Western Ontario and McMaster Universities Osteoarthritis Index ( WOMAC ) score'],
                    'stp_words': get_stop_words(),
                    'key_words_keep':['valu', 'score', 'time', 'level', 'scale', 'test'],
                    'relative_terms':['total', 'average', 'increase', 'decrease', 'negative', 'positive']
                         }

    def xml_wrapper(self):
        main_dir = 'adding_tags_to_ebm/'
        data_dir = os.path.abspath(os.path.join(main_dir, 'aggregated'))
        if not os.path.exists(os.path.dirname(data_dir)):
            os.makedirs(os.path.dirname(data_dir))

        sub_dir = os.path.abspath(os.path.join(data_dir, 'test'))
        if not os.path.exists(os.path.dirname(sub_dir)):
            os.makedirs(os.path.dirname(sub_dir))
        t = pd.DataFrame()
        outcomes = []

        turker, ebm_extract = e.read_anns('hierarchical_labels', 'outcomes', \
                                                    ann_type='aggregated', model_phase='test/gold')
        outcomes.clear()
        for pmid, doc in ebm_extract.items():
            for lst in e.print_labeled_spans_2(doc):
                for tup in lst:
                    outcomes.append(tup)

        outcomes_df = pd.DataFrame(outcomes)
        outcomes_df.columns = ['Label', 'Outcomes']

        t = pd.concat([t, outcomes_df], axis=1)

        return t

    def pos_co_occurrence_cleaning(self):
        cleaned_labels_outcomes = []
        pos_tagged = os.path.abspath(os.path.join('adding_tags_to_ebm/aggregated/train', 'stanford_pos_tagged_train.json'))
        with open(pos_tagged, 'r') as tg_json:
            tagged_file = json.load(tg_json)
            i = 0
            for sent in tagged_file:
                text_wrds = sent['text']
                text_pos = sent['pos']
                label = sent['label']

                split_text_pos, split_text_wrds = text_pos.split(), text_wrds.split()

                split_text_wrds_copy = [stem_word(i) for i in split_text_wrds]

                for i in [stem_word(h) for h in self.unwanted['relative_terms']]:
                    if i in split_text_wrds_copy:
                        e = split_text_wrds_copy.index(i)
                        del split_text_wrds[e]
                        del split_text_pos[e]

                for j in range(len(split_text_pos)):
                    if j == 0:
                        if split_text_pos[j] in (['TO','IN','CD','CC',',']):
                            split_text_wrds = split_text_wrds[1:]
                            split_text_pos = split_text_pos[1:]

                text_wrds = ' '.join([i for i in split_text_wrds])
                text_pos = ' '.join([i for i in split_text_pos])

                #scenario one (outcomes with a single word)
                if len(split_text_wrds) == 1:
                    if text_wrds in get_punctuation():
                        pass
                    elif stem_word(text_wrds.lower()) in self.unwanted['rct_instruments_results']+self.unwanted['key_words_keep']:
                        pass
                    else:
                        cleaned_labels_outcomes.append((label, text_wrds.strip()))

                # sceanrio four (outcome with key words)
                elif any(stem_word(i).lower() in self.unwanted['key_words_keep'] for i in split_text_wrds):
                    f = ' '.join([stem_word(i) for i in split_text_wrds])
                    to_add = ''
                    for j in self.unwanted['key_words_keep']:
                        if re.search('{}$'.format(j), f):
                            if j == 'level':
                                if re.search('^((VBN\sIN)|(NNS\sIN))', text_pos):
                                    to_add = remove_first(split_text_wrds, 2)
                                    cleaned_labels_outcomes.append((label, to_add.strip()))
                                else:
                                    to_add = text_wrds
                                    cleaned_labels_outcomes.append((label, to_add.strip()))
                            elif j == 'score':
                                print(text_wrds)
                                print(text_pos)
                #                 to_add = ' '.join(i for i in split_text_wrds[:-1])
                #                 cleaned_labels_outcomes.append((label, to_add.strip()))
                #             elif j == 'scale':
                #                 to_add = ' '.join(i for i in split_text_wrds[:-1])
                #                 cleaned_labels_outcomes.append((label, to_add.strip()))
                #             elif j == 'valu':
                #                 to_add = ' '.join(i for i in split_text_wrds[:-3])
                #                 cleaned_labels_outcomes.append((label, to_add.strip()))
                #             else:
                #                 to_add = text_wrds
                #                 cleaned_labels_outcomes.append((label, to_add.strip()))
                #         elif re.search('^{}'.format(j), f):
                #             to_add = text_wrds
                #             cleaned_labels_outcomes.append((label, to_add.strip()))
                #         elif re.search(' {} '.format(j), f):
                #             if re.search('JJ CC JJ', text_pos):
                #                 to_add = text_wrds
                #                 cleaned_labels_outcomes.append((label, to_add.strip()))
                #             else:
                #                 for x, y in zip(re.split(' and | or | and$| or$|,|^and |^or ', text_wrds),
                #                                 re.split('CC|,', text_pos)):
                #                     if y.strip() != 'JJ':
                #                         to_add = x.strip()
                #                         if to_add:
                #                             if to_add.split()[0].lower() in self.unwanted['relative_terms']:
                #                                 to_add = remove_first(to_add, 1)
                #                             cleaned_labels_outcomes.append((label, to_add.strip()))
                #
                # # sceanrio 2 (outcomes with a couple of words)
                # elif len(split_text_wrds) == 2:
                #     if any(i in get_punctuation() for i in split_text_pos):
                #         for i in get_punctuation():
                #             text_wrds = text_wrds.replace(i, '').strip()
                #         if text_wrds:
                #             cleaned_labels_outcomes.append((label, text_wrds))
                #     elif any(i == 'CC' or i == 'DT' or i == 'TO' for i in split_text_pos):
                #         s = [i for i,j in zip(split_text_wrds, split_text_pos) if j not in ['TO', 'CC', 'DT']]
                #         to_add = str(s).strip("'[]'")
                #         if to_add.lower() not in self.unwanted['rct_instruments_results']:
                #             if stem_word(to_add.lower()) not in self.unwanted['key_words_keep']:
                #                 cleaned_labels_outcomes.append((label, to_add.strip()))
                #     else:
                #         cleaned_labels_outcomes.append((label, text_wrds.strip()))

                # # sceanrio 3 (outcomes with three words)
                # elif len(split_text_wrds) == 3:
                #     print(text_wrds)
                #     print(text_pos)
                #     to_add = ''
                #     if all(i == ',' or i == 'CC' or i == 'DT' for i in split_text_pos):
                #         cleaned_labels_outcomes.append((label, ''))
                #     elif any(i == ',' or i == 'CC' or i == 'DT' for i in split_text_pos):
                #         if re.search('^DT|^CC|^,', text_pos):
                #             for y in split_text_pos:
                #                 if y == 'DT' or y == 'CC' or y == ',' or y == '':
                #                     split_text_wrds = split_text_wrds[1:]
                #                 else:
                #                     break
                #             text_wrds = remove_first(split_text_wrds, 0)
                #             cleaned_labels_outcomes.append((label, text_wrds.strip()))
                #         elif re.search('CC$|DT$|,$', text_pos):
                #             cleaned_labels_outcomes.append((label, ' '.join(i for i in split_text_wrds[:-1])))
                #         else:
                #             if re.search('CC|,', text_pos):
                #                 cleaned_labels_outcomes.append((label, ' '.join(i for i in split_text_wrds[:1])))
                #                 cleaned_labels_outcomes.append((label, ' '.join(i for i in split_text_wrds[2:])))
                #             else:
                #                 cleaned_labels_outcomes.append((label, text_wrds.strip()))
                #     else:
                #         to_add = text_wrds.strip()
                #         cleaned_labels_outcomes.append((label, to_add))




                # #scenario five: only nouns
                # elif (all(i.__contains__('NN') for i in split_text_pos)):
                #     if text_wrds:
                #         if text_wrds.split()[0] in self.unwanted['relative_terms']:
                #             text_wrds = remove_first(text_wrds, 1)
                #         cleaned_labels_outcomes.append((label, text_wrds.strip()))
                #
                # #scenario six: nouns and conjunctions
                # elif all(i.__contains__('NN') or i == 'CC' for i in split_text_pos):
                #     if(len([j for j in split_text_pos if j == 'CC']) == 1):
                #         for x in re.split(' and | or | and$| or$', text_wrds):
                #             if x.split()[0].lower() in self.unwanted['relative_terms']:
                #                 x = remove_first(x, 1)
                #             cleaned_labels_outcomes.append((label, x.strip()))
                #     elif(len([j for j in split_text_pos if j == 'CC']) > 1):
                #         cleaned_labels_outcomes.append((label, ' '.join(text_wrds.split()[-2:]).strip()))
                #
                #
                # # scenario seven: nouns and injunctions
                # elif all(i.__contains__('NN') or i == 'IN' for i in split_text_pos):
                #     to_add = ''
                #     if re.search('^(NN.{0,1}\sIN)', text_pos):
                #         split_text_wrds = split_text_wrds[2:]
                #         if split_text_wrds[0].lower() in self.unwanted['relative_terms']:
                #             to_add = remove_first(split_text_wrds, 1)
                #         else:
                #             to_add = remove_first(split_text_wrds, 0)
                #     else:
                #         to_add = text_wrds
                #     if to_add:
                #         cleaned_labels_outcomes.append((label, to_add.strip()))
                #
                #
                # # scenario eight: only nouns and adjectives
                # elif all(i.__contains__('NN') or i == 'JJ' for i in split_text_pos):
                #     if split_text_wrds[0].lower() in self.unwanted['relative_terms']:
                #         text_wrds = remove_first(text_wrds, 1)
                #         cleaned_labels_outcomes.append((label, text_wrds.strip()))
                #     else:
                #         cleaned_labels_outcomes.append((label, text_wrds.strip()))
                #
                # # scenario nine: only conjunctions and adjectives
                # elif all(i.__contains__('CC') or i == 'JJ' for i in split_text_pos):
                #     pass
                #
                # # scenario ten: only nouns, conjunctions, and injunctions
                # elif all(i.__contains__('NN') or i == 'CC' or i == 'IN' for i in split_text_pos):
                #     to_add = ''
                #     if re.search('^(NN.{0,1}\sIN)', text_pos):
                #         to_add = remove_first(split_text_wrds, 2)
                #         if re.search('(IN\sCC\sIN\sNN.{0,1})$', text_pos):
                #             cleaned_labels_outcomes.append((label, to_add))
                #         else:
                #             for x in re.split(' and | or | ; | and$| or$| ;$', to_add):
                #                 cleaned_labels_outcomes.append((label, x))
                #     else:
                #         if re.search('((IN|NN.{0,1})\sCC\s[(NN.{0,1})(IN)\s]*NN.{0,1})$', text_pos):
                #             cleaned_labels_outcomes.append((label, text_wrds))
                #         else:
                #             cleaned_labels_outcomes.append((label, ' '.join(text_wrds.split()[-3:])))
                #
                # # scenario eleven: only nouns, conjunctions, and adjectives
                # elif all(i.__contains__('NN') or i == 'CC' or i == 'JJ' for i in split_text_pos):
                #     sc5_split_index = split_text_pos.index('CC')
                #     if split_text_wrds[sc5_split_index-1] in split_text_wrds[sc5_split_index+1:]:
                #         cleaned_labels_outcomes.append((label, ' '.join(i for i in split_text_wrds[:sc5_split_index])))
                #         cleaned_labels_outcomes.append((label, ' '.join(i for i in split_text_wrds[(sc5_split_index+1):])))
                #     else:
                #         cleaned_labels_outcomes.append((label, text_wrds.strip()))
                #
                # #scenario twelve: nouns and comma's
                # elif all(i.__contains__('NN') or i == ',' for i in split_text_pos):
                #     for i in text_wrds.split(','):
                #         cleaned_labels_outcomes.append((label, i.strip()))
                #
                # #scenario thirteen: nouns or commas and conjunctions
                # elif all(i.__contains__('NN') or i == ',' or i == 'CC' for i in split_text_pos):
                #     for i in re.split(',| and | or | ; | and$| or$| ;$', text_wrds):
                #         if i:
                #             cleaned_labels_outcomes.append((label, i.strip()))
                #
                # #scenario fourteen: nouns and commas and injunctions
                # elif all(i.__contains__('NN') or i == ',' or i == 'IN' for i in  split_text_pos):
                #     for i in re.split(',', text_wrds):
                #         if i is not None:
                #             cleaned_labels_outcomes.append((label, i.strip()))
                #
                # # scenario fifteen
                # elif all(i.__contains__('NN') or i == ',' or i == 'JJ' for i in  split_text_pos):
                #     pass
                #
                # #scenario sixteen nouns, comma's, injunctions and adjectives
                # elif all(i.__contains__('NN') or i == ',' or i == 'IN' or i == 'JJ' for i in split_text_pos):
                #     if re.search('^(NN.{0,1}\sIN)', text_pos):
                #         text_wrds = remove_first(text_wrds, 2)
                #         cleaned_labels_outcomes.append((label, text_wrds.strip()))
                #     else:
                #         cleaned_labels_outcomes.append((label, text_wrds.strip()))
                #
                # # scenario seventeen nouns, injunctions, commas and conjunctions
                # elif all(i.__contains__('NN') or i == ',' or i == 'IN'  or i == 'CC' for i in split_text_pos):
                #     if re.search('(NN\s[,\sCC]+NN.{0,1}\s(IN)\sNN.{0,1})$', text_pos):
                #         cleaned_labels_outcomes.append((label, ' '.join([i for i in split_text_wrds[-6:]]).strip()))
                #     else:
                #         for x, y in zip(re.split(' and | or | and$| or$|,|^and |^or ', text_wrds), re.split('CC|,|;', text_pos)):
                #             if y.strip() != 'NNS':
                #                 cleaned_labels_outcomes.append((label, x.strip()))
                #
                # # scenario eighteen: nouns, injunctions, conjunctions and adjectives
                # elif all(i.__contains__('NN') or i == 'IN' or i == 'CC' or i == 'JJ' for i in split_text_pos):
                    # for x, y in zip(re.split(' and | or | and$| or$|,|^and |^or ', text_wrds), re.split('CC|,|;', text_pos)):
                    #     if re.search('^(NN.{0,1}\sIN)', y):
                    #         x = remove_first(x, 2)
                    #         if x.split()[0].lower() in self.unwanted['relative_terms']:
                    #             x = remove_first(x, 1)
                    #         else:
                    #             x = x
                    #     else:
                    #         if x.split()[0].lower() in self.unwanted['relative_terms']:
                    #             x = remove_first(x, 1)
                    #         else:
                    #             x = x
                    #     cleaned_labels_outcomes.append((label, x.strip()))

            #     # scenario ninteen: nouns, injunctions, conjunctions, adjectives and commas
            #     elif all(i.__contains__('NN') or i == 'IN' or i == 'CC' or i == 'JJ' or i == ',' for i in split_text_pos):
            #         for x in re.split(' and | or | and$| or$|,|^and |^or ', text_wrds):
            #             cleaned_labels_outcomes.append((label, x.strip()))
            #
            #     # scenario twenty: nouns and verbs
            #     elif all(i.__contains__('NN') or i.__contains__('V')  for i in split_text_pos):
            #         split_text_wrds = ['' if j.__contains__('V') else i for i,j in zip(split_text_wrds, split_text_pos)]
            #         if split_text_wrds[0] in self.unwanted['relative_terms'] or split_text_wrds[0] == '':
            #             text_wrds = remove_first(split_text_wrds, 1)
            #         cleaned_labels_outcomes.append((label, text_wrds.strip()))
            #
            #     # scenario twenty one: nouns, verbs and conjunctions
            #     elif all(i.__contains__('NN') or i.__contains__('V') or i == 'CC' for i in split_text_pos):
            #         text_wrds = ' '.join([i for i in ['' if j == 'VB' else i for i, j in zip(split_text_wrds, split_text_pos)]])
            #         if re.search('((VBG|NN.{0,1})\sCC\s(NN.{0,1}|VBG)+\s(NN.{0,1})+)$', text_pos):
            #             cleaned_labels_outcomes.append((label, text_wrds.strip()))
            #         else:
            #             for x in re.split(' and | or | and$| or$|,|^and |^or ', text_wrds):
            #                 cleaned_labels_outcomes.append((label, x.strip()))
            #
            #     #scenario twenty two: nouns, verbs and injunctions
            #     elif all(i.__contains__('NN') or i.__contains__('V') or i == 'IN' for i in split_text_pos):
            #         if re.search('^(NN.{0,1}\sIN)', text_pos):
            #             text_wrds = remove_first(split_text_wrds, 2)
            #             cleaned_labels_outcomes.append((label, text_wrds.strip()))
            #         else:
            #             cleaned_labels_outcomes.append((label, text_wrds.strip()))
            #
            #     # scenario twenty three: nouns, verbs and adjectives
            #     elif all(i.__contains__('NN') or i.__contains__('V') or i == 'JJ' for i in split_text_pos):
            #         if re.search('^VB', text_pos):
            #             text_wrds = remove_first(text_wrds, 1)
            #             cleaned_labels_outcomes.append((label, text_wrds.strip()))
            #         else:
            #             cleaned_labels_outcomes.append((label, text_wrds.strip()))
            #
            #     # scenario twenty four: nouns, verbs and commas
            #     elif all(i.__contains__('NN') or i.__contains__('V') or i == ',' for i in split_text_pos):
            #         pass
            #
            #     # scenario twenty five: nouns, verbs, commas and injunctions
            #     elif all(i.__contains__('NN') or i.__contains__('V') or i == 'CC' or i == 'IN' for i in split_text_pos):
            #         for x in re.split(' and | or | and$| or$|,|^and |^or ', text_wrds):
            #             cleaned_labels_outcomes.append((label, x.strip()))
            #
            #     # scenario twenty six: nouns, verbs, conjunctions and adjectives
            #     elif all(i.__contains__('NN') or i.__contains__('V') or i == 'CC' or i == 'JJ' for i in split_text_pos):
            #         for x, y in zip(re.split(' and | or | and$| or$|,|^and |^or ', text_wrds), re.split('CC|,|;', text_pos)):
            #             if y.strip() != 'JJ':
            #                 cleaned_labels_outcomes.append((label, x.strip()))
            #
            #     # scenario twenty seven: nouns, verbs, conjunctions and acommas
            #     elif all(i.__contains__('NN') or i.__contains__('V') or i == 'CC' or i == ',' for i in split_text_pos):
            #         pass
            #
            #     # scenario twenty seven: nouns, verbs, injunctions and adjectives
            #     elif all(i.__contains__('NN') or i.__contains__('V') or i == 'IN' or i == 'JJ' for i in split_text_pos):
            #         split_text_wrds = ['' if j == 'VBN' else i for i, j in zip(split_text_wrds, split_text_pos)]
            #         if re.search('^(NN.{0,1}\sIN)|^(VB.{0,1}\sIN)', text_pos):
            #             text_wrds = remove_first(text_wrds, 2)
            #             cleaned_labels_outcomes.append((label, text_wrds.strip()))
            #         else:
            #             text_wrds = remove_first(split_text_wrds, 0)
            #             cleaned_labels_outcomes.append((label, text_wrds.strip()))
            #
            #     # scenario twenty eight: nouns, verbs, injunctions,d adjectives and conjunctions
            #     elif all(i.__contains__('NN') or i.__contains__('V') or i == 'IN' or i == 'JJ' or i == 'CC' for i in split_text_pos):
            #         if re.search('(VB.{0,1}\sCC\sJJ\sNN.{0,1}\sIN\sNN.{0,1})$', text_pos):
            #             cleaned_labels_outcomes.append((label, text_wrds.strip()))
            #         else:
            #             for x in re.split(' and | or | and$| or$|,|^and |^or ', text_wrds):
            #                 cleaned_labels_outcomes.append((label, x.strip()))
            #
            #     # scenario twenty nine: nouns, verbs, injunctions,d adjectives, conjunctions and commas
            #     elif all(i.__contains__('NN') or i.__contains__('V') or i == 'IN' or i == 'JJ' or i == 'CC' or i == ',' for i in split_text_pos):
            #         text_wrds = ' '.join([i for i in ['' if j == 'VBN' or j == 'VB' else i for i, j in zip(split_text_wrds, split_text_pos)]])
            #         tex_pos =  ' '.join([i for i in ['' if j == 'VBN' or j == 'VB' else j for j in  split_text_pos]])
            #         for x,y in zip(re.split(' and | or | and$| or$|,|^and |^or ', text_wrds), re.split(' CC |,', text_pos)):
            #             if re.search('^(NN.{0,1}\sIN)', y):
            #                 x = remove_first(x, 2)
            #                 cleaned_labels_outcomes.append((label, x.strip()))
            #             else:
            #                 cleaned_labels_outcomes.append((label, x.strip()))
            #
            #     # scenario twenty nine: nouns and braces
            #     elif all(i.__contains__('NN') or i == '(' or i == ')' or i == '[' or i == ']' for i in split_text_pos):
            #         cleaned_labels_outcomes.append((label, text_wrds.strip()))
            #
            #     # scenario thirty one: nouns, adjectives and braces
            #     elif all(i.__contains__('NN') or i == 'JJ' or i == '(' or i == ')' or i == '[' or i == ']' for i in split_text_pos):
            #         cleaned_labels_outcomes.append((label, text_wrds.strip()))
            #
            #     # scenario thirty two: nouns, injunctions and braces
            #     elif all(i.__contains__('NN') or i == 'IN' or i == '(' or i == ')' or i == '[' or i == ']' for i in split_text_pos):
            #         if re.search('(^\(.*\)$)', text_wrds):
            #             text_wrds = text_wrds.strip("()")
            #             if re.search('^(NN.{0,1}\sIN)', text_pos):
            #                 text_wrds = remove_first(split_text_wrds, 2)
            #                 cleaned_labels_outcomes.append((label, text_wrds.strip()))
            #             else:
            #                 cleaned_labels_outcomes.append((label, text_wrds.strip()))
            #         else:
            #             if re.search('^(NN.{0,1}\sIN)', text_pos):
            #                 text_wrds = remove_first(split_text_wrds, 2)
            #                 cleaned_labels_outcomes.append((label, text_wrds.strip()))
            #             else:
            #                 cleaned_labels_outcomes.append((label, text_wrds.strip()))
            #
            #     # scenario thirty three: nouns, commas and braces
            #     elif all(i.__contains__('NN') or i == ',' or i == '(' or i == ')' or i == '[' or i == ']' for i in split_text_pos):
            #         pass
            #
            #     # scenario thirty four: nouns, conjunctions and braces
            #     elif all(i.__contains__('NN') or i == 'CC' or i == '(' or i == ')' or i == '[' or i == ']' for i in split_text_pos):
            #         pass
            #
            #     # scenario thirty five: nouns, conjunctions, adjectives and braces
            #     elif all(i.__contains__('NN') or i == 'CC' or i == 'JJ'  or i == '(' or i == ')' or i == '[' or i == ']' for i in  split_text_pos):
            #         for x in re.split(' and | or | and$| or$|,|^and |^or ', text_wrds):
            #             cleaned_labels_outcomes.append((label, x.strip()))
            #
            #     # scenario thirty six: nouns, injunctions, adjectives and braces
            #     elif all(i.__contains__('NN') or i == 'JJ' or i == 'IN' or i == '(' or i == ')' or i == '[' or i == ']' for i in split_text_pos):
            #         cleaned_labels_outcomes.append((label, text_wrds.strip()))
            #
            #     # scenario thirty seven: nouns, conjunctions, injunctions, adjectives and braces
            #     elif all(i.__contains__('NN') or i == 'CC' or i == 'IN' or i == 'JJ' or i == '(' or i == ')' or i == '[' or i == ']' for i in split_text_pos):
            #         pass
            #
            #     # scenario thirty eight: nouns,  adjectives and braces
            #     elif all(i.__contains__('NN') or i == 'JJ' or i == '(' or i == ')' or i == '[' or i == ']' for i in split_text_pos):
            #         pass
            #
            #     # scenario thirty nine: nouns, commas,  adjectives and braces
            #     elif all(i.__contains__('NN') or i == 'JJ' or i == ',' or i == '(' or i == ')' or i == '[' or i == ']' for i in split_text_pos):
            #         text_wrds = (re.sub('(\(|\[|\)|\])', ',', text_wrds))
            #         for x in re.split(',+', text_wrds):
            #             cleaned_labels_outcomes.append((label, x.strip()))
            #
            #     # scenario forty: nouns, injunctions, commas and braces
            #     elif all(i.__contains__('NN')  or i == 'IN' or i == ',' or i == '(' or i == ')' or i == '[' or i == ']' for i in split_text_pos):
            #         for x,y in zip(re.split(',+', text_wrds), re.split(',+', text_pos)):
            #             if re.search('^(NN.{0,1}\sIN)', y):
            #                 x = remove_first(x, 2)
            #                 cleaned_labels_outcomes.append((label, x.strip()))
            #             else:
            #                 cleaned_labels_outcomes.append((label, x.strip()))
            #
            #     # scenario forty one: nouns, conjunctions, injunctions, commas and braces
            #     elif all(i.__contains__('NN') or i == 'CC' or i == 'IN' or i == 'JJ' or i == ',' or i == '(' or i == ')' or i == '[' or i == ']' for i in split_text_pos):
            #         text_wrds = (re.sub('(\(|\[|\)|\])', ',', text_wrds))
            #         for x in re.split(' and | or | and$| or$|,|^and |^or ', text_wrds):
            #             if re.search('^(NN.{0,1}\sIN)', text_pos):
            #                 x = remove_first(x, 2)
            #                 cleaned_labels_outcomes.append((label, x.strip()))
            #             else:
            #                 cleaned_labels_outcomes.append((label, x.strip()))
            #
            #     # scenario forty two: JJR's
            #     elif any(i.__contains__('JJR')for i in split_text_pos):
            #         text_wrds = ' '.join([i for i in ['' if j == 'JJR' or j == 'VBN' or j == 'RBR' else i for i, j in zip(split_text_wrds, split_text_pos)]]).strip()
            #         text_pos =  ' '.join([i for i in ['' if j.__contains__('JJR') or j == 'VBN' or j == 'RBR' else j for j in split_text_pos]]).strip()
            #         text_wrds = re.sub('(\(|\[|\)|\])', ',', text_wrds)
            #         for x,y in zip(re.split(' and | or | and$| or$|,|^and |^or ', text_wrds), re.split(',|CC', text_pos)):
            #             if re.search('^(NN.{0,1}\sIN)', y):
            #                 x = remove_first(x, 2)
            #                 cleaned_labels_outcomes.append((label, x.strip()))
            #             else:
            #                 cleaned_labels_outcomes.append((label, x.strip()))
            #
            #     # scenario forty two: Adverbs
            #     elif any(i.__contains__('RB') for i in split_text_pos):
            #         if re.search('^(NN.{0,1}\sIN)', text_pos):
            #             text_wrds = remove_first(text_wrds, 2)
            #             cleaned_labels_outcomes.append((label, text_wrds.strip()))
            #         else:
            #             cleaned_labels_outcomes.append((label, text_wrds.strip()))
            #
            #     # scenario forty three: nouns, DT's and Injunctions
            #     elif all(i.__contains__('NN') or i == 'DT' or i =='IN' for i in split_text_pos):
            #         cleaned_labels_outcomes.append((label, text_wrds.strip()))
            #
            #     elif all(i.__contains__('NN') or i == 'DT' or i == 'JJ' for i in split_text_pos):
            #         pass
            #
            #     elif all(i.__contains__('NN') or i == 'DT' or i == 'JJ' or i == ',' for i in split_text_pos):
            #         pass
            #
            #     elif all(i.__contains__('NN') or i == 'DT' or i.__contains__('V') or i == 'IN' for i in split_text_pos):
            #         pass
            #
            #     # scenario forty three: starting with nouns then injunction
            #     elif re.search('^(NN.{0,1})+\sIN', text_pos):
            #         if re.search('^(NN.{0,1}\sIN\sDT)', text_pos):
            #             text_wrds = remove_first(split_text_wrds, 3)
            #             text_wrds = re.sub('(\(|\[|\)|\])', ',', text_wrds)
            #
            #             text_pos = ' '.join([i for i in [',' if j == '(' or j == ')' or j == '[' or j == ']' else j for j in split_text_pos[3:]]]).strip()
            #
            #             for x, y in zip(re.split(' and | or | and$| or$|,|^and |^or ', text_wrds), re.split(',|CC', text_pos)):
            #                 if y.strip() != 'JJ':
            #                     if x != '':
            #                         if x.split()[0].strip() in self.unwanted['relative_terms']:
            #                             x = remove_first(x, 1)
            #                             cleaned_labels_outcomes.append((label, x.strip()))
            #                         else:
            #                             cleaned_labels_outcomes.append((label, x.strip()))
            #                 else:
            #                     cleaned_labels_outcomes.append((label, x.strip()))
            #         else:
            #             if re.search('^(NN.{0,1}\sIN)', text_pos):
            #                 if re.search('^(NN.{0,1}\sIN\sPRP)|^(NN.{0,1}\sIN\sVB)', text_pos):
            #                     cleaned_labels_outcomes.append((label, text_wrds.strip()))
            #                 else:
            #                     text_wrds = remove_first(split_text_wrds, 2)
            #                     if re.search('(IN CC IN DT NN)$', text_pos):
            #                         cleaned_labels_outcomes.append((label, text_wrds.strip()))
            #                     else:
            #                         for i in re.split(' and | or | and$| or$|,|^and |^or ', text_wrds):
            #                             cleaned_labels_outcomes.append((label, i.strip()))
            #             else:
            #                 cleaned_labels_outcomes.append((label, text_wrds.strip()))
            #
            #     # scenario forty four: couns, numbers, commas, adjectives, conjunctions, DT's
            #     elif all(i.__contains__('NN') or i=='CD' or i == 'IN' or i == ',' or i == 'JJ' or i == 'CC' or i == 'DT' for i in split_text_pos):
            #         if re.search('(CD\sCC\sNN)$', text_pos):
            #             if split_text_wrds[0] in self.unwanted['relative_terms']:
            #                 text_wrds = remove_first(split_text_wrds, 1)
            #             cleaned_labels_outcomes.append((label, text_wrds.strip()))
            #         else:
            #             for i in re.split(' and | or | and$| or$|,|^and |^or ', text_wrds):
            #                 if i:
            #                     if i.split()[0] in self.unwanted['relative_terms']:
            #                         i = remove_first(i, 1)
            #                     cleaned_labels_outcomes.append((label, i.strip()))
            #
            #     # scenario forty five: nouns, DT's, TO's, commas, adjectives, conjunctions, DT's
            #     elif all(i.__contains__('NN') or i == 'TO' or i == 'IN'  or i == 'CC' or i == ',' or i == 'JJ' or i ==  'DT' for i in split_text_pos):
            #         cleaned_labels_outcomes.append((label, text_wrds.strip()))
            #
            #     else:
            #         for p, q in zip([w.strip() for w in text_wrds.split(',')], [h.strip() for h in text_pos.split(',')]):
            #             if (re.search('^(CC\sIN)|^(VB\sDT)', q)):
            #                 p = remove_first(p, 2)
            #                 cleaned_labels_outcomes.append((label, p.strip()))
            #             else:
            #                 p = re.sub('\(','-', p)
            #                 p = re.sub('\)', '',p)
            #                 if re.search('^(\-|\[|(CC))', q):
            #                     p = remove_first(p, 1)
            #                     cleaned_labels_outcomes.append((label, p.strip()))
            #                 elif q == 'DT' or q == 'CD' or q == 'JJ' or len(p) < 2:
            #                     pass
            #                 elif len([i for i in q.split() if i.__contains__('J')]) > len([i for i in q.split() if i.__contains__('N')]):
            #                     cleaned_labels_outcomes.append((label, p.strip()))
            #                 elif re.search('^NN\sIN', q):
            #                     to_add = remove_first(p, 2)
            #                     if to_add:
            #                         cleaned_labels_outcomes.append((label, to_add.strip()))
            #                     else:
            #                         cleaned_labels_outcomes.append((label, p.strip()))
            #                 else:
            #                     cleaned_labels_outcomes.append((label, p.strip()))
            #
            # cleaned_labels_outcomes = [(i[0], re.sub(r'\(\s*\)', '', i[1])) for i in cleaned_labels_outcomes]
            # cleaned_labels_outcomes = [(i[0], re.sub(r'\s+', ' ', i[1])) for i in cleaned_labels_outcomes]
            #
            # _train_tagged_dir = os.path.abspath(os.path.join('cleaned_outcomes', 'train'))
            # if not os.path.exists(_train_tagged_dir):
            #     os.makedirs(_train_tagged_dir)
            #
            # testing_df = pd.DataFrame(cleaned_labels_outcomes)
            # testing_df.columns = ['Label', 'Outcome']
            # testing_df = testing_df[testing_df['Outcome'] != '']
            #
            # testing_df.to_csv(os.path.join(_train_tagged_dir, 'train_set.csv'))
            # print(tabulate(testing_df, headers='keys', tablefmt='psql'))

def remove_first(x, n):
    if type(x) == str:
        if n >= 0:
            x = ' '.join(i for i in x.split()[n:])
        else:
            x = ' '.join(i for i in x.split()[:n])
    elif type(x) == list:
        if n >= 0:
            x = ' '.join(i for i in x[n:])
        else:
            x = ' '.join(i for i in x[:n])
    return x

#phase two of cleaning up, phrases where puncutation is hunging e.g. bone marrow , offset are turned to bone marrow, offset, and un-necessary punctuation at the start of sentences is eliminated
def sub_span(outcome_string):
    pattern_y = re.compile(r'(.\s[.,:;!?%])')
    fix_string = pattern_y.findall(outcome_string)
    if fix_string:
        for j in fix_string:
            outcome_string = outcome_string.replace(j, str(j[0]+j[2]))
    else:
        outcome_string = outcome_string
    outcome_string = re.sub(r'^[\s.,;:!?\)\]]', '', outcome_string)
    return outcome_string

#concatenate all the frames extracted from the different annotations
def final_label_outcome(ann_type=[]):
    list_of_frames = []
    stat_lexicon = extracting_statistical_lexicon()
    for ann in ann_type:
        locate_dir = os.path.abspath(os.path.join('adding_tags_to_ebm', ann))
        csv_files = [f for f in glob(os.path.join(locate_dir, '*.csv')) if os.path.basename(f).__contains__('and')]

        for file in csv_files:
            f = pd.read_csv(file)
            f = f[['Label','Outcomes']]
            list_of_frames.append(f)

    concat_frames = pd.concat(list_of_frames)
    concat_frames_sorted = concat_frames.sort_values(by='Label')

    concat_frames_sorted.drop_duplicates(subset=['Label','Outcomes'], keep=False)
    #concat_frames_sorted['Outcome'] = concat_frames_sorted['Outcome'].apply(lambda x: sub_span(x))
    concat_frames_sorted = concat_frames_sorted.loc[concat_frames_sorted['Outcomes'].str.len() > 1]
    concat_frames_sorted.to_csv(os.path.join('./adding_tags_to_ebm/', 'and_or_comma_colon_outcomes.csv'))

def extracting_statistical_lexicon(other_sources):
    path = './statistical_terms_RCT.pdf'
    statistical_terms = []
    stp_wrds = get_stop_words()
    stat_file = PyPDF2.PdfFileReader(path)
    pages = stat_file.getNumPages()
    with open('stat_file.txt', 'w') as stat_:
        for i in range(pages):
            contents = stat_file.getPage(i).extractText()
            c = StringIO(contents).readlines()
            for i in c:
                d = re.split('\s{2}', i)[0]
                d = re.split('\n', d)
                d = re.findall('^\s[\w\-\s]+', d[0])
                if d:
                    d_1 = re.sub('[0-9]+', '', d[0]).strip()
                    if len(d_1) > 2:
                        if d_1[0] != '-':
                            d_2 = d_1.split()
                            if len([f for f in d_2 if f in stp_wrds]) > 1:
                                d_3 = ''
                            else:
                                d_3 = ' '.join([i for i in d_2])
                            if d_3 != '':
                                d_3 = re.sub('\s\-\s','-',d_3)
                                d_3 = re.sub('Þ ','fi',d_3)
                                if d_3.lower() not in ['toxicity', 'the']:
                                    statistical_terms.append(d_3)

        _stats = [i.lower() for i in (statistical_terms + other_sources)]
        for term in list(set(_stats)):
            stat_.write('{}\n'.format(term))
    return list(set(_stats))


def scrap_stat_terms(scrap_source = []):
    scrapped_terms = []
    for url in scrap_source:
        content = req.get(url, stream=True)
        if content.status_code:
            if content.headers['Content-Type'].lower().find('html'):
                needed_content = BeautifulSoup(content.content, 'html.parser')
                for term in needed_content.select('tr'):
                    scrapped_terms.append(term.text)

    return scrapped_terms


def visulize_statistical_term_occurrence(x, stat_terms):
    outcomes_str = ' '.join([i for i in x])
    outcomes_str = [i for i in outcomes_str.split() if i.lower() in stat_terms]
    outcomes_str = ' '.join([i for i in outcomes_str])
    most_common_statistical_terms = WordCloud(background_color='white', height=400, width=600).generate_from_text(outcomes_str)
    plt.title('Most commonly used statistical terms')
    plt.imshow(most_common_statistical_terms)
    plt.axis('off')
    plt.savefig('stat.png')
    plt.show()

def get_stop_words():
    stp_wrds = list(set(stopwords.words('english')))
    return stp_wrds

def get_punctuation():
    return list(string.punctuation)

def stem_word(x):
    stem = PorterStemmer()
    return stem.stem(x)

def check_english(x):
    is_english = enchant.Dict()
    return is_english.check(x)

#fix the randomly positioned items in words inside squared or curly braces
def fix_mispositioned_words_in_braces(x):
    pattern_x = re.compile(r'[\[\(]\s+[\w\s,;:%/._+-]+\s+[\)\]]')
    irregular_string = pattern_x.findall(x)
    if irregular_string:
        for item in irregular_string:
            regularised_string = item[0] + item[2:-2] + item[-1]
            x = x.replace(item, regularised_string)
    return


def outcomes_seperated_by_and_or_comma_colon(pair):
    out = []
    k = ['and', ', and', 'or', ', or', ',', ' ,', ';', ' ;']
    if (pair[1] not in k):
        pair_1 = pair[1].split()
        if all(i in get_stop_words() for i in pair_1) == False:
            if 'and' in pair_1 or ',' in pair_1 or 'or' in pair_1 or ';' in pair_1:
                return pair

if __name__=='__main__':
    run = annotate_text()
    #df = run.xml_wrapper()
    #run.df_frame(df)
    run.pos_co_occurrence_cleaning()
