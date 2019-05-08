import os
import pandas as pd
import copy
from nltk.tokenize import word_tokenize
import ebm_nlp_demo as e
import re
from glob import glob
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords, wordnet
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
import spacy

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
        self.nlp = spacy.load("en_core_web_sm")

        self.properties = {'annotators': 'pos', 'outputFormat': 'json'}
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
                                                    '()',
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
                                                    ann_type='aggregated', model_phase='train')
        outcomes.clear()
        for pmid, doc in ebm_extract.items():
            for lst in e.print_labeled_spans_2(doc):
                for tup in lst:
                    outcomes.append(tup)

        outcomes_df = pd.DataFrame(outcomes)
        outcomes_df.columns = ['Label', 'Outcomes']

        t = pd.concat([t, outcomes_df], axis=1)
        return t

    def df_frame(self, df):
        o_phrase = {}
        tagged = []
        #with open('tagged_train.json', 'w') as tagger:
        for label, out_come in zip(df['Label'], df['Outcomes']):
            words_postags = []
            #remove un-necessary random punctuation
            for i in (self.unwanted['punctuation']):
                out_come = re.sub(re.escape('{}'.format(i)), '', out_come, flags=re.IGNORECASE)

            #remove unwanted key words
            for i in (self.unwanted['rct_instruments_results'] + self.unwanted['relative_terms']):
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

                out_come = ' '.join(i for i in out_come).strip()

                # extract part of speech for words in outcome
                for elem in word_tokenize(out_come):
                    if (elem in ['(', ')', '[', ']']):
                        words_postags.append((elem, elem))
                    elif re.search(r'\+|\-', elem):
                        words_postags.append((elem, 'NN'))
                    elif re.search(r'\/', elem):
                        pos_tagger = self.nlp(elem)
                        pos_tagger_len = len([i for i in pos_tagger])
                        if pos_tagger_len == 1:
                            w = str(pos_tagger).strip('[]')
                            if w in ['CD', 'CC']:
                                words_postags.append((elem, w))
                            elif re.search('^/\w+', elem):
                                words_postags.append((elem[0], 'NN'))
                                words_postags.append((elem[1:], 'NN'))
                        else:
                            words_postags.append((elem, 'NN'))
                    else:
                        pos_tagger_else = self.nlp(elem)
                        for tok in pos_tagger_else:
                            words_postags.append((tok.text, tok.tag_))

                # #check to make sure the recent changes haven't introduced an unwanted starting Parts of speech, remove verb, adverb, conjunction or preoposition
                # for i in words_postags:
                #     if words_postags[0][1].__contains__('RB') or words_postags[0][1] == 'IN' or words_postags[0][1] == 'CC':
                #         words_postags = words_postags[1:]
                #     elif words_postags[0][1].__contains__('V'):
                #         stemed_pos = nltk.pos_tag([stem_word(words_postags[0][0])])
                #         if stemed_pos[0][1].__contains__('NN'):
                #             words_postags = words_postags[0:]
                #         else:
                #             words_postags = words_postags[1:]
                #     else:
                #         break
                #
                # text_pos = ' '.join(i[1] for i in words_postags).strip()
                # text_wrds = ' '.join(i[0] for i in words_postags).strip()
                #
                # o_phrase['label'] = label
                # o_phrase['text'] = text_wrds
                # o_phrase['pos'] = text_pos
                #
                # tagged.append(o_phrase.copy())

            #json.dump(tagged, tagger, indent=4, sort_keys=True)

def stem_word(x):
    stem = PorterStemmer()
    return stem.stem(x)

def synonym_set(x):
    _synset = []
    for i in x:
        s = wordnet.synsets(i)
        for j in s:
            for w in j.lemmas():
                _synset.append(w.name())
    return list(set(_synset))

def get_stop_words():
    stp_wrds = list(set(stopwords.words('english')))
    return stp_wrds

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

def medpost_train():
    r = os.listdir(os.path.abspath('.'))
    fnames = glob(os.path.join(os.path.abspath('medpost'), 'tagged', '*.ioc'))

    Train_data =[]
    for i in fnames:
        f = open(i, 'r').readlines()
        for j in f:
            if len(j) != 13 and not re.search('^(P\d{1,4})', j):
                print(j)
                # e = j.split()
                # k,tags=[],[]
                # for t in e:
                #     k.clear()
                #     tags.clear()
                #     q = t.split('_')
                #     k.append(q[0])
                #     tags.append(q[1])
                # Train_data.append((' '.join(i for i in k), {"tags":tags}))








if __name__=='__main__':
    # run = annotate_text()
    # df = run.xml_wrapper()
    # print(tabulate(df, headers='keys', tablefmt='psql'))
    #run.df_frame(df)
    medpost_train()

