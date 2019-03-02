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


pd.set_option('display.max_rows', 50)

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

        self.unwanted = {'punctuation': ['.', "'", ';', ':'],
                    'stat_lexicon': extracting_statistical_lexicon(scrap_stat_terms(scrap_source=[])),
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
                    'key_words_keep':['valu', 'score', 'time', 'level', 'scale', 'test']
                         }

    def xml_wrapper(self):
        main_dir = 'adding_tags_to_ebm/'
        data_dir = os.path.abspath(os.path.join(main_dir, 'aggregated'))
        if not os.path.exists(os.path.dirname(data_dir)):
            os.makedirs(os.path.dirname(data_dir))

        sub_dir = os.path.abspath(os.path.join(data_dir, 'test'))
        if not os.path.exists(sub_dir):
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

        #t.to_csv(os.path.abspath(os.path.join(sub_dir, 'final_train.csv')))
                #retrieving outcomes seperated by and or or or comma or semi-colons
        #         outcom = list(map(lambda x:outcomes_seperated_by_and_or_comma_colon(x), list(set(outcomes))))
        #         outcom = [i for i in outcom if i is not None]
        #         for i in outcom:
        #             and_or_comma_colon_outcomes.append(i)
        #
        # and_or_comma_colon_outcomes_df = pd.DataFrame(outcomes)
        # and_or_commt = pd.DataFrame(outcomes)a_colon_outcomes_df.columns = ['Label', 'Outcomes']

        #print(tabulate(t, headers='keys', tablefmt='psql'))
        return t

    def df_frame(self, df):
        cleaned_labels_outcomes = []
        for label, out_come in zip(df['Label'], df['Outcomes']):
            words_postags = []
            #remove un-necessary random punctuation
            for i in (self.unwanted['punctuation']):
                out_come = re.sub(re.escape('{}'.format(i)), '', out_come, flags=re.IGNORECASE)

            #remove unwanted key words
            for i in (self.unwanted['rct_instruments_results']):
                p_utwd = ' {} '.format(i)
                p_utwd_2 = '^{} '.format(i)
                p_utwd_3 = ' {}$'.format(i)
                if i.lower() == out_come.lower():
                    out_come = out_come.replace(out_come, '')
                elif re.search(p_utwd, out_come, re.IGNORECASE):
                    out_come = re.sub(p_utwd, ' ', out_come, flags=re.IGNORECASE)
                elif re.search(p_utwd_2, out_come, re.IGNORECASE):
                    out_come = re.sub(p_utwd_2, '', out_come, flags=re.IGNORECASE).strip()
                elif re.search(p_utwd_3, out_come, re.IGNORECASE):
                    out_come = re.sub(p_utwd_3, '', out_come, flags=re.IGNORECASE)

            #eliminating the statistical terms from the corpus
            for i in self.unwanted['stat_lexicon']:
                p_utwd_4 = '^{} '.format(i)
                p_utwd_5 = ' {} '.format(i)
                p_utwd_6 = ' {}$'.format(i)

                if i == out_come.lower():
                    cleaned_labels_outcomes.append((label, out_come))
                    out_come = out_come.replace(out_come, '')
                else:
                    if not out_come.lower().__contains__('side effects'):
                        if re.search(p_utwd_4, out_come, re.IGNORECASE):
                            out_come = re.sub(p_utwd_4, '', out_come, flags=re.IGNORECASE).strip()
                        elif re.search(p_utwd_5, out_come, re.IGNORECASE):
                            out_come = re.sub(p_utwd_5, ' ', out_come, flags=re.IGNORECASE)
                        elif re.search(p_utwd_6, out_come, re.IGNORECASE):
                            out_come = re.sub(re.escape(p_utwd_6), '', out_come, flags=re.IGNORECASE)

            if out_come:
                #split the outcome
                out_come = out_come.split()

                # ensure first and last elements are neither stopwords nor punctuations
                l = len(out_come)-1
                out_come[0] = '' if out_come[0].lower() in self.unwanted['stp_words'] or out_come[0] in self.unwanted['punctuation'] else out_come[0]
                out_come[l] = '' if out_come[l].lower() in self.unwanted['stp_words'] or out_come[l] in self.unwanted['punctuation']  else out_come[l]

                out_come = (' '.join(i for i in out_come)).strip()

                #extract part of speech for words in outcome
                for elem in word_tokenize(out_come):
                    if(elem in ['(',')','[',']']):
                        words_postags.append((elem, elem))
                    elif(elem.__contains__('+')):
                        words_postags.append((elem, 'NN'))
                    else:
                        pos_tagger = self.stan.annotate(elem, properties=self.properties)
                        words_postags.append((pos_tagger['sentences'][0]['tokens'][0]['word'], pos_tagger['sentences'][0]['tokens'][0]['pos']))

                #check to make sure the recent changes haven't introduced an unwanted starting Parts of speech, remove verb, adverb, conjunction or preoposition

                for i in words_postags:
                    if words_postags[0][1].__contains__('RB') or words_postags[0][1] == 'IN' or words_postags[0][1] == 'CC':
                        words_postags = words_postags[1:]
                    elif words_postags[0][1].__contains__('V'):
                        stemed_pos = nltk.pos_tag([stem_word(words_postags[0][0])])
                        if stemed_pos[0][1].__contains__('NN'):
                            words_postags = words_postags[0:]
                        else:
                            words_postags = words_postags[1:]
                    else:
                        break


                text_pos = ' '.join(i[1].replace(i[1], 'NN') if i[0].__contains__('-') or i[0].__contains__('+') else i[1] for i in words_postags).strip()
                text_wrds = ' '.join(i[0] for i in words_postags).strip()
                split_text_pos, split_text_wrds = text_pos.split(), text_wrds.split()

                if len(split_text_wrds) == 1:
                    if text_wrds in get_punctuation():
                        cleaned_labels_outcomes.append('')
                    elif stem_word(text_wrds.lower()) in self.unwanted['rct_instruments_results']+self.unwanted['key_words_keep']:
                        cleaned_labels_outcomes.append('')
                    else:
                        cleaned_labels_outcomes.append((label, text_wrds))

                elif len(split_text_wrds) == 2:
                    if all(i == ',' or i == 'CC' or i == 'DT' for i in split_text_pos):
                        cleaned_labels_outcomes.append('')
                    elif any(i == ',' or i == 'CC' or i == 'DT' for i in split_text_pos):
                        s = [i for i,j in zip(split_text_wrds, split_text_pos) if j not in [',', 'CC', 'DT']]
                        to_add = str(s).strip("'[]'")
                        if to_add.lower() not in self.unwanted['rct_instruments_results']:
                            if stem_word(to_add.lower()) not in self.unwanted['key_words_keep']:
                                cleaned_labels_outcomes.append(to_add)
                    else:
                        cleaned_labels_outcomes.append((label, text_wrds))

                elif len(split_text_wrds) == 3:
                    if all(i == ',' or i == 'CC' or i == 'DT' for i in split_text_pos):
                        cleaned_labels_outcomes.append('')
                    elif any(i == ',' or i == 'CC' or i == 'DT' for i in split_text_pos):
                        if re.search('^DT|^CC|^,', text_pos):
                            cleaned_labels_outcomes.append((label, ' '.join(i for i in split_text_wrds[1:])))
                        elif re.search('CC$|DT$|,$', text_pos):
                            cleaned_labels_outcomes.append((label, ' '.join(i for i in split_text_wrds[:-1])))
                        else:
                            if re.search('CC|,', text_pos):
                                cleaned_labels_outcomes.append(split_text_wrds[0])
                                cleaned_labels_outcomes.append(split_text_wrds[2])
                            else:
                                cleaned_labels_outcomes.append(text_wrds)
                                
                elif any(stem_word(i).lower() in self.unwanted['key_words_keep'] for i in split_text_wrds):
                    f = ' '.join([stem_word(i) for i in split_text_wrds])
                    for j in self.unwanted['key_words_keep']:
                        if re.search('{}$'.format(j), f):
                            if j == 'level':
                                if re.search('^((VBN\sIN)|(NNS\sIN))', text_pos):
                                    cleaned_labels_outcomes.append((label,  ' '.join(i for i in split_text_wrds[2:])))
                                else:
                                    cleaned_labels_outcomes.append((label, text_wrds))
                            elif j =='score':
                                cleaned_labels_outcomes.append((label, ' '.join(i for i in split_text_wrds[:-1])))
                            elif j =='scale':
                                cleaned_labels_outcomes.append((label, ' '.join(i for i in split_text_wrds[:-1])))
                            elif j == 'value':
                                cleaned_labels_outcomes.append((label, ' '.join(i for i in split_text_wrds[:-3])))
                            else:
                                cleaned_labels_outcomes.append((label, text_wrds))
                        elif re.search('^{}'.format(j), f):
                            cleaned_labels_outcomes.append((label, text_wrds))
                        elif re.search(' {} '.format(j), f):
                            if re.search('JJ CC JJ', text_pos):
                                cleaned_labels_outcomes.append((label, text_wrds))
                            else:
                                for x, y in zip(re.split(' and | or | and$| or$|,|^and |^or ', text_wrds), re.split('CC|,', text_pos)):
                                    if y != 'JJ':
                                        cleaned_labels_outcomes.append((label, x))

                #scenario one: ALL phrases are nouns
                elif (all(i.__contains__('NN') for i in split_text_pos)):
                    if text_wrds:
                        cleaned_labels_outcomes.append((label, text_wrds))

                #scenario 2: only one conjunction
                elif all(i.__contains__('NN') or i == 'CC' for i in split_text_pos):
                    if(len([j for j in split_text_pos if j == 'CC']) == 1):
                        for x in re.split('and |or |; |, | and$| or$| ,$| ;$', text_wrds):
                            cleaned_labels_outcomes.append((label, x))
                    elif(len([j for j in split_text_pos if j == 'CC']) > 1):
                        cleaned_labels_outcomes.append((label, ' '.join(text_wrds.split()[-2:])))

                # scenario 2: only one conjunction
                elif all(i.__contains__('NN') or i == 'IN' for i in split_text_pos):
                    if re.search('^(NN.{0,1}\sIN)', text_pos):
                        cleaned_labels_outcomes.append((label,' '.join([i for i in split_text_wrds[2:]])))
                    else:
                        cleaned_labels_outcomes.append((label, text_wrds))

                # scenario 3: only one conjunction and one injunction
                elif all(i.__contains__('NN') or i == 'IN' or i == 'CC' for i in split_text_pos):
                    if re.search('^(NN.{0,1}\sIN)', text_pos):
                        to_add = ' '.join([i for i in split_text_wrds[2:]])
                        if re.search('(IN\sCC\sIN\sNN.{0,1})$', text_pos):
                            cleaned_labels_outcomes.append((label, to_add))
                        else:
                            for x in re.split('and |or |; |, | and$| or$| ,$| ;$', to_add):
                                cleaned_labels_outcomes.append((label, x))
                    else:
                        if re.search('((IN|NN.{0,1})\sCC\s[(NN.{0,1})(IN)\s]*NN.{0,1})$', text_pos):
                            cleaned_labels_outcomes.append((label, text_wrds))
                        else:
                            cleaned_labels_outcomes.append((label, ' '.join(text_wrds.split()[-3:])))

                #scenario 4 and adjective and a conjunctions
                elif all(i.__contains__('NN') or i == 'JJ' for i in split_text_pos):
                    cleaned_labels_outcomes.append((label, text_wrds))

                # scenario 5
                elif all(i.__contains__('CC') or i == 'JJ' for i in split_text_pos):
                    pass

                # scenario 6
                elif all(i.__contains__('NN') or i == 'JJ' or i == 'CC' for i in split_text_pos):
                    sc5_split_index = split_text_pos.index('CC')
                    if split_text_wrds[sc5_split_index-1] in split_text_wrds[sc5_split_index+1:]:
                        cleaned_labels_outcomes.append((label, ' '.join(i for i in split_text_wrds[:sc5_split_index])))
                        cleaned_labels_outcomes.append((label, ' '.join(i for i in split_text_wrds[(sc5_split_index+1):])))
                    else:
                        cleaned_labels_outcomes.append((label, text_wrds))

                #scenario 7
                elif all(i.__contains__('NN') or i == ',' for i in split_text_pos):
                    for i in text_wrds.split(','):
                        cleaned_labels_outcomes.append((label, i))

                #scenario 8
                elif all(i.__contains__('NN') or i == ',' or i == 'CC' for i in split_text_pos):
                    for i in re.split(', |and |or |; ', text_wrds):
                        if i is not None:
                            cleaned_labels_outcomes.append((label, i))

                #scenario 9
                elif all(i.__contains__('NN') or i == ',' or i == 'IN' for i in  split_text_pos):
                    for i in re.split(',', text_wrds):
                        if i is not None:
                            cleaned_labels_outcomes.append((label, i))

                # scenario 10
                elif all(i.__contains__('NN') or i == ',' or i == 'JJ' for i in  split_text_pos):
                    pass

                #scenario 11 double check
                elif all(i.__contains__('NN') or i == ',' or i == 'IN' or i == 'JJ' for i in split_text_pos):
                    if re.search('^(NN.{0,1}\sIN)', text_pos):
                        cleaned_labels_outcomes.append((label, ' '.join([i for i in split_text_wrds[2:]])))
                    else:
                        cleaned_labels_outcomes.append((label, text_wrds))

                # scenario 12
                elif all(i.__contains__('NN') or i == ',' or i == 'IN'  or i == 'CC' for i in split_text_pos):
                    if re.search('(NN\s,\sCC\sNN.{0,1}\s(IN)\sNN.{0,1})$', text_pos):
                        cleaned_labels_outcomes.append((label, ' '.join([i for i in split_text_wrds[-6:]])))
                        for i in re.split(',', ' '.join([i for i in split_text_wrds[-6:]])):
                            cleaned_labels_outcomes.append((label, i))
                    else:
                        for i in re.split(',| and ', text_wrds):
                            cleaned_labels_outcomes.append((label, i))

                # scenario 13 double check
                elif all(i.__contains__('NN') or i == 'IN' or i == 'CC' or i == 'JJ' for i in split_text_pos):
                    cleaned_labels_outcomes.append((label, text_wrds))

                # scenario 13 double check
                elif all(i.__contains__('NN') or i == 'IN' or i == 'CC' or i == 'JJ' or i == ',' for i in split_text_pos):
                    for x in re.split(' and | or | and$| or$|,|^and |^or ', text_wrds):
                        cleaned_labels_outcomes.append((label, x))

                # scenario 14 double check
                elif all(i.__contains__('NN') or i.__contains__('V')  for i in split_text_pos):
                    text_wrds = ['' if j.__contains__('V') else i for i,j in zip(split_text_wrds, split_text_pos)]
                    cleaned_labels_outcomes.append((label, ' '.join([i for i in split_text_wrds])))

                # scenario 14 double check
                elif all(i.__contains__('NN') or i.__contains__('V') or i == 'CC' for i in split_text_pos):
                    text_wrds = ' '.join([i for i in ['' if j == 'VB' else i for i, j in zip(split_text_wrds, split_text_pos)]])
                    if re.search('((VBG|NN.{0,1})\sCC\s(NN.{0,1}|VBG)+\s(NN.{0,1})+)$', text_pos):
                        cleaned_labels_outcomes.append((label,text_wrds))
                    else:
                        for x in re.split(' and | or | and$| or$|,|^and |^or ', text_wrds):
                            cleaned_labels_outcomes.append((label, x))

                #scenario 15 double check
                elif all(i.__contains__('NN') or i.__contains__('V') or i == 'IN' for i in split_text_pos):
                    if re.search('^(NN.{0,1}\sIN)', text_pos):
                        cleaned_labels_outcomes.append((label, ' '.join([i for i in split_text_wrds[2:]])))
                    else:
                        cleaned_labels_outcomes.append((label, text_wrds))

                # scenario 16 double check
                elif all(i.__contains__('NN') or i.__contains__('V') or i == 'JJ' for i in split_text_pos):
                    if re.search('^VB', text_pos):
                        cleaned_labels_outcomes.append((label, ' '.join([i for i in split_text_wrds[1:]])))
                    else:
                        cleaned_labels_outcomes.append((label, text_wrds))

                # scenario 17 double check
                elif all(i.__contains__('NN') or i.__contains__('V') or i == ',' for i in split_text_pos):
                    pass

                # scenario 18 double check
                elif all(i.__contains__('NN') or i.__contains__('V') or i == 'CC' or i == 'IN' for i in split_text_pos):
                    for x in re.split(' and | or | and$| or$|,|^and |^or ', text_wrds):
                        cleaned_labels_outcomes.append((label, x))

                # scenario 18 double check
                elif all(i.__contains__('NN') or i.__contains__('V') or i == 'CC' or i == 'JJ' for i in split_text_pos):
                    if len([i for i in split_text_pos if i == 'CC']) > 1:
                        cleaned_labels_outcomes.append((label, text_wrds))
                    else:
                        for x in re.split(' and | or | and$| or$|,|^and |^or ', text_wrds):
                            cleaned_labels_outcomes.append((label, x))

                # scenario 19 double check
                elif all(i.__contains__('NN') or i.__contains__('V') or i == 'CC' or i == ',' for i in split_text_pos):
                    pass

                # scenario 20 double check
                elif all(i.__contains__('NN') or i.__contains__('V') or i == 'IN' or i == 'JJ' for i in split_text_pos):
                    text_wrds = ['' if j == 'VBN' else i for i, j in zip(split_text_wrds, split_text_pos)]
                    if re.search('^(NN.{0,1}\sIN)|^(VB.{0,1}\sIN)', text_pos):
                        cleaned_labels_outcomes.append((label, ' '.join([i for i in text_wrds[2:]])))
                    else:
                        cleaned_labels_outcomes.append((label, text_wrds))

                # scenario 19 double check
                elif all(i.__contains__('NN') or i.__contains__('V') or i == 'IN' or i == 'JJ' or i == 'CC' for i in split_text_pos):
                    if re.search('(VB.{0,1}\sCC\sJJ\sNN.{0,1}\sIN\sNN.{0,1})$', text_pos):
                        cleaned_labels_outcomes.append((label, text_wrds))
                    else:
                        for x in re.split(' and | or | and$| or$|,|^and |^or ', text_wrds):
                            cleaned_labels_outcomes.append((label, x))

                elif all(i.__contains__('NN') or i.__contains__('V') or i == 'IN' or i == 'JJ' or i == 'CC' or i == ',' for i in split_text_pos):
                    text_wrds = ' '.join([i for i in ['' if j == 'VBN' or j == 'VB' else i for i, j in zip(split_text_wrds, split_text_pos)]])
                    tex_pos =  ' '.join([i for i in ['' if j == 'VBN' or j == 'VB' else j for j in  split_text_pos]])
                    for x,y in zip(re.split(' and | or | and$| or$|,|^and |^or ', text_wrds), re.split(' CC |,', text_pos)):
                        if re.search('^(NN.{0,1}\sIN)', y):
                            cleaned_labels_outcomes.append((label, ' '.join([i for i in x[2:]])))
                        else:
                            cleaned_labels_outcomes.append((label, x))

                # scenario 21 double check
                elif all(i.__contains__('NN') or i == '(' or i == ')' or i == '[' or i == ']' for i in split_text_pos):
                    cleaned_labels_outcomes.append((label, text_wrds))

                elif all(i.__contains__('NN') or i == 'JJ' or i == '(' or i == ')' or i == '[' or i == ']' for i in split_text_pos):
                    cleaned_labels_outcomes.append((label, text_wrds))

                elif all(i.__contains__('NN') or i == 'IN' or i == '(' or i == ')' or i == '[' or i == ']' for i in split_text_pos):
                    if re.search('(^\(.*\)$)', text_wrds):
                        text_wrds = text_wrds.strip("()")
                        if re.search('^(NN.{0,1}\sIN)', text_pos):
                            cleaned_labels_outcomes.append((label, ' '.join([i for i in split_text_wrds[2:]])))
                        else:
                            cleaned_labels_outcomes.append((label, text_wrds))
                    else:
                        if re.search('^(NN.{0,1}\sIN)', text_pos):
                            cleaned_labels_outcomes.append((label, ' '.join([i for i in split_text_wrds[2:]])))
                        else:
                            cleaned_labels_outcomes.append((label, text_wrds))

                elif all(i.__contains__('NN') or i == ',' or i == '(' or i == ')' or i == '[' or i == ']' for i in split_text_pos):
                    pass

                elif all(i.__contains__('NN') or i == 'CC' or i == '(' or i == ')' or i == '[' or i == ']' for i in split_text_pos):
                    pass
                elif all(i.__contains__('NN') or i == 'CC' or i == 'JJ'  or i == '(' or i == ')' or i == '[' or i == ']' for i in  split_text_pos):
                    for x in re.split(' and | or | and$| or$|,|^and |^or ', text_wrds):
                        cleaned_labels_outcomes.append((label, x))


                elif all(i.__contains__('NN') or i == 'JJ' or i == 'IN' or i == '(' or i == ')' or i == '[' or i == ']' for i in split_text_pos):
                    cleaned_labels_outcomes.append((label, text_wrds))


                elif all(i.__contains__('NN') or i == 'CC' or i == 'IN' or i == 'JJ' or i == '(' or i == ')' or i == '[' or i == ']' for i in split_text_pos):
                    pass


                elif all(i.__contains__('NN') or i == 'JJ' or i == '(' or i == ')' or i == '[' or i == ']' for i in split_text_pos):
                    pass

                elif all(i.__contains__('NN') or i == 'JJ' or i == ',' or i == '(' or i == ')' or i == '[' or i == ']' for i in split_text_pos):
                    text_wrds = (re.sub('(\(|\[|\)|\])', ',', text_wrds))
                    for x in re.split(',+', text_wrds):
                        cleaned_labels_outcomes.append((label, x))

                elif all(i.__contains__('NN')  or i == 'IN' or i == ',' or i == '(' or i == ')' or i == '[' or i == ']' for i in split_text_pos):
                    text_wrds = (re.sub('(\(|\[|\)|\])', ',', text_wrds))
                    tex_pos = ' '.join([i for i in [',' if j == '(' or j == ')' else j for j in split_text_pos]])
                    for x,y in zip(re.split(',', text_wrds), re.split(',', text_pos)):
                        if re.search('^(NN.{0,1}\sIN)', y):
                            cleaned_labels_outcomes.append((label, ' '.join([i for i in x[2:]])))
                        else:
                            cleaned_labels_outcomes.append((label, x))
                elif all(i.__contains__('NN') or i == 'CC' or i == 'IN' or i == 'JJ' or i == ',' or i == '(' or i == ')' or i == '[' or i == ']' for i in split_text_pos):
                    text_wrds = (re.sub('(\(|\[|\)|\])', ',', text_wrds))
                    for x in re.split(',', text_wrds):
                        if re.search('^(NN.{0,1}\sIN)', text_pos):
                            cleaned_labels_outcomes.append((label, ' '.join([i for i in split_text_wrds[2:]])))
                        else:
                            cleaned_labels_outcomes.append((label, text_wrds))


                elif any(i.__contains__('JJR')for i in split_text_pos):
                    text_wrds = ' '.join([i for i in ['' if j == 'JJR' or j == 'VBN' or j == 'RBR' else i for i, j in zip(split_text_wrds, split_text_pos)]]).strip()
                    text_pos =  ' '.join([i for i in ['' if j.__contains__('JJR') or j == 'VBN' or j == 'RBR' else j for j in split_text_pos]]).strip()
                    text_wrds = re.sub('(\(|\[|\)|\])', ',', text_wrds)

                    for x,y in zip(re.split(' and | or | and$| or$|,|^and |^or ', text_wrds), re.split(',|CC', text_pos)):
                        if re.search('^(NN.{0,1}\sIN)', y):
                            cleaned_labels_outcomes.append((label, ' '.join([i for i in x.split()[2:]])))
                        else:
                            cleaned_labels_outcomes.append((label, x))

                elif any(i.__contains__('RB') for i in split_text_pos):
                    if re.search('^(NN.{0,1}\sIN)', text_pos):
                        cleaned_labels_outcomes.append((label, ' '.join([i for i in text_wrds.split()[2:]])))
                    else:
                        cleaned_labels_outcomes.append((label, text_wrds))

                elif all(i.__contains__('NN') or i == 'DT' or i =='IN' for i in split_text_pos):
                    cleaned_labels_outcomes.append((label, text_wrds))

                elif all(i.__contains__('NN') or i == 'DT' or i == 'JJ' for i in split_text_pos):
                    pass

                elif all(i.__contains__('NN') or i == 'DT' or i == 'JJ' or i == ',' for i in split_text_pos):
                    pass

                elif all(i.__contains__('NN') or i == 'DT' or i.__contains__('V') or i == 'IN' for i in split_text_pos):
                    pass

                elif re.search('^(NN.{0,1})+\sIN', text_pos):
                    if re.search('^(NN.{0,1}\sIN\sDT)', text_pos):
                        text_wrds = ' '.join([i for i in split_text_wrds[3:]])
                        text_wrds = re.sub('(\(|\[|\)|\])', ',', text_wrds)

                        text_pos = ' '.join([i for i in [',' if j == '(' or j == ')' or j == '[' or j == ']' else j for j in split_text_pos[3:]]]).strip()

                        for x, y in zip(re.split(' and | or | and$| or$|,|^and |^or ', text_wrds), re.split(',|CC', text_pos)):
                            if y != 'JJ':
                                cleaned_labels_outcomes.append((label, x))
                            else:
                                cleaned_labels_outcomes.append((label, x))
                    else:
                        if re.search('^(NN.{0,1}\sIN)', text_pos):
                            if re.search('^(NN.{0,1}\sIN\sPRP)|^(NN.{0,1}\sIN\sVB)', text_pos):
                                cleaned_labels_outcomes.append((label, text_wrds))
                            else:
                                text_wrds = ' '.join(i for i in split_text_wrds[2:])
                                if re.search('(IN CC IN DT NN)$', text_pos):
                                    cleaned_labels_outcomes.append((label, text_wrds))
                                else:
                                    for i in re.split(' and | or | and$| or$|,|^and |^or ', text_wrds):
                                        cleaned_labels_outcomes.append((label, i))

                elif all(i.__contains__('NN') or i=='CD' or i == 'IN' or i == ',' or i == 'JJ' or i == 'CC' or i == 'DT' for i in split_text_pos):
                    if re.search('(CD\sCC\sNN)$', text_pos):
                        cleaned_labels_outcomes.append((label, text_wrds))
                    else:
                        for i in re.split(' and | or | and$| or$|,|^and |^or ', text_wrds):
                            cleaned_labels_outcomes.append((label, i))


                elif all(i.__contains__('NN') or i == 'TO' or i == 'IN'  or i == 'CC' or i == ',' or i == 'JJ' or i ==  'DT' for i in split_text_pos):
                    cleaned_labels_outcomes.append((label, text_wrds))

                else:
                    for p, q in zip([w.strip() for w in text_wrds.split(',')], [h.strip() for h in text_pos.split(',')]):
                        if (re.search('^(CC\sIN)|^(VB\sDT)', q)):
                            cleaned_labels_outcomes.append((label, ' '.join(i for i in p.strip()[2:])))
                        else:
                            p = re.sub('\(','-', p)
                            p = re.sub('\)', '',p)
                            if re.search('^(\-|\[|(CC))', q):
                                p = ' '.join(i for i in p.split()[1:])
                            else:
                                cleaned_labels_outcomes.append((label, p))
                        # elif q == 'DT' or q == 'CD' or q == 'JJ' or len(p) < 2:
                        #     pass
                        # elif len([i for i in q.split() if i.__contains__('J')]) > len([i for i in q.split() if i.__contains__('N')]):
                        #     cleaned_labels_outcomes.append((label, p))
                        # elif re.search('^NN\sIN', q):
                        #     to_add = (' '.join(i for i in p.split()[2:]))
                        #     if to_add:
                        #         cleaned_labels_outcomes.append((label, to_add))
                        #     else:
                        #         cleaned_labels_outcomes.append((label, p))


        #         if (len([j for j in split_text_pos if j == 'CC']) == 1):
        #             x = re.search(r'^(NN.{0,2})+\sIN\s(NN.{0,2})\sCC\s(NN.{0,2})', text_pos)
        #             x1 = re.search(r'^(NN.{0,2})+\sCC\s.+', text_pos)
        #             if x is not None:
        #                 if x.group()[0:6] == 'NNS IN' and stem_word(text_wrds.split()[0]):
        #                     for x in re.split('and|or|;', text_wrds):
        #                         cleaned_labels_outcomes.append((label, x))
        #                 else:
        #                     cleaned_labels_outcomes.append((label, text_wrds))
        #             elif x1 is not None:
        #                 if all(i == 'NN' or i == 'IN' or i == 'CC' for i in split_text_pos):
        #                     cleaned_labels_outcomes.append((label, text_wrds))
        #                 else:
        #                     sc4_split_index = split_text_pos.index('IN')
        #                     phrase_6 = ' '.join(i for i in split_text_wrds[(sc4_split_index+1):])
        #                     cleaned_labels_outcomes.append((label, phrase_6))
        #             else:
        #                 t = re.compile('(IN\sCC\sIN\sNN)$')
        #                 if t.search(text_pos):
        #                     phrase_7 = ' '.join(i for i in split_text_wrds[:-4])
        #                     cleaned_labels_outcomes.append((label, phrase_7))
        #                 else:
        #                     cleaned_labels_outcomes.append((label, text_wrds))
        #         else:
        #             cleaned_labels_outcomes.append((label, text_wrds))
        #




        #     elif any(i == ',' for i in split_text_wrds):
        #         for u,v in zip(re.split(',+| and ', text_wrds), re.split(',+| CC ', text_pos)):
        #             if v.__contains__('V'):
        #                 for i in re.split('was|measured| by', u):
        #                     if i:
        #                         cleaned_labels_outcomes.append((label,i))
        #             else:
        #                 cleaned_labels_outcomes.append(((label, u)))
        #     else:
        #         if text_pos.__contains__('FW'):
        #             cleaned_labels_outcomes.append((label, text_wrds))
        #         elif text_pos.__contains__('V'):
        #             for u in (split_text_pos):
        #                 if split_text_pos[0].__contains__('V') or split_text_pos[0].__contains__('R'):
        #                     split_text_wrds = split_text_wrds[1:]
        #             cleaned_labels_outcomes.append((label, ' '.join(i for i in split_text_wrds)))
        #         elif re.search('IN\sDT\sNN\sIN', text_pos):
        #             for u in re.split(' with a reduction in ', text_wrds):
        #                 cleaned_labels_outcomes.append((label, u))
        #         elif re.search('(CD\sCC\sNN)$', text_pos):
        #             cleaned_labels_outcomes.append((label, text_wrds))
        #         else:
        #             for u in re.split('and', text_wrds):
        #                 cleaned_labels_outcomes.append((label, u))
        #
        #
        # cleaned_labels_outcomes_dict = dict(cleaned_labels_outcomes)
        # df_frame = pd.DataFrame(cleaned_labels_outcomes)
        # df_frame.columns = ['Label','Outcome']
        # df_frame.to_csv('and_or_corrected.csv')
        #print(tabulate(df_frame, headers='keys', tablefmt='psql'))

        # new_dir = os.path.abspath('adding_tags_to_ebm/aggregated')
        # if not os.path.exists(new_dir):
        #     os.makedirs(new_dir)
        # df_frame.to_csv(os.path.join(new_dir, 'aggregated_test_crowd.csv'))



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
    df = run.xml_wrapper()
    run.df_frame(df)

