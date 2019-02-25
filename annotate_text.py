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
        self.turker, self.ebm_extract = e.read_anns('hierarchical_labels', 'outcomes', \
                                      ann_type = 'aggregated', model_phase = 'test/gold')
        self.stan = StanfordCoreNLP('http://localhost:9000')

        self.properties = {'annotators': 'pos',  'outputFormat': 'json'}

        self.unwanted = {'punctuation': ['.', "'", ';', ':'],
                    'stat_lexicon': extracting_statistical_lexicon()+['specificity','sensitivity','correlation'],
                    'rct_instruments_results': ['subjective significance questionnaire',
                                                'Baseline tumour marker',
                                                'Stroop Test and Wisconsin Card Sorting Test',
                                                'visual analogue scale ','questionnaire',
                                                'Stroop Test','Wisconsin Card Sorting Test','age', 'gender', 'bmi'],
                    'stp_words': get_stop_words(),
                    'key_words_keep':['valu', 'score', 'time', 'level', 'scale', 'test']
                         }

    def xml_wrapper(self):
        main_dir = 'adding_tags_to_ebm/'
        data_dir = os.path.abspath(os.path.join(main_dir, 'aggregated', 'train_spans.txt'))
        if not os.path.exists(os.path.dirname(data_dir)):
            os.makedirs(os.path.dirname(data_dir))
        print('Tagged spans are being written in file location %s'%(data_dir))
        lab_outcome = []
        lab_outcome.clear()
        #context_file = open('lack_context.txt', 'w')

        with open(data_dir, 'w') as new_f:
            orig, annon = [], []
            and_or_comma_colon_outcomes = []

            for pmid, doc in self.ebm_extract.items():
                outcomes = []
                for lst in e.print_labeled_spans_2(doc):
                    for tup in lst:
                        outcomes.append(tup)
                #outcomes_ordered_by_len = list(set(sorted(outcomes, key=lambda x: len(x[1]), reverse=True)))
                abs_art = ' '.join([i for i in word_tokenize(doc.text)])
                abs_art_clone = copy.copy(abs_art)
                new_f.write('{}\n\n'.format(abs_art))

                #retrieving outcomes seperated by and or or or comma or semi-colons
                outcom = list(map(lambda x:outcomes_seperated_by_and_or_comma_colon(x), list(set(outcomes))))
                outcom = [i for i in outcom if i is not None]
                for i in outcom:
                    and_or_comma_colon_outcomes.append(i)

            #     for pair in list(set(outcomes)):
            #         x = str(pair[1])
            #         y = str(Outcomes[pair[0]])
            #         x_tokenized = x.split()
            #         x_ann = []
            #         for i in x_tokenized:
            #             if i in self.unwanted['punctuation'] or i in self.unwanted['stat_lexicon']:
            #                 x_ann.append(str(Outcomes['No label']))
            #             else:
            #                 x_ann.append(y)
            #         x_ann_append = ' '.join([i for i in x_ann])
            #         if abs_art_clone.__contains__(x):
            #             if (x not in self.unwanted['stat_lexicon']):
            #                 abs_art_clone = abs_art_clone.replace(x, x_ann_append)
            #     for i,j in zip(abs_art.split(), abs_art_clone.split()):
            #         orig.append(i)
            #         annon.append(j)
            # xo = pd.DataFrame({'Original':orig})
            # xa = pd.DataFrame({'Annonymous': annon})
            # xl = pd.concat([xo, xa], axis=1)

        and_or_comma_colon_outcomes_df = pd.DataFrame(and_or_comma_colon_outcomes)
        and_or_comma_colon_outcomes_df.columns = ['Label', 'Outcomes']
        # and_or_comma_colon_outcomes_df.to_csv('and_or_comma_colon_outcomes_gold.csv')

        return and_or_comma_colon_outcomes_df

    def df_frame(self, df):
        cleaned_labels_outcomes = []
        _df = pd.DataFrame()
        v = 0
        for label, out_come in zip(df['Label'], df['Outcomes']):
            words_postags = []
            #remove un-necessary random punctuation and un_wanted key words
            for i in (self.unwanted['punctuation'] + self.unwanted['rct_instruments_results']):
                out_come = re.sub(re.escape('{} '.format(i)), '', out_come, flags=re.IGNORECASE)

            # #split the outcome
            out_come = out_come.split()

            # ensure first and last elements are neither stopwords nor punctuations
            l = len(out_come)-1
            out_come[0] = '' if out_come[0].lower() in self.unwanted['stp_words'] or out_come[0] in self.unwanted['punctuation'] else out_come[0]
            out_come[l] = '' if out_come[l].lower() in self.unwanted['stp_words'] or out_come[l] in self.unwanted['punctuation']  else out_come[l]

            #eliminate statistical terms
            #out_come = ['' if i.lower() in self.unwanted['stat_lexicon'] else i for i in out_come]
            out_come = [i for i in out_come if i.lower() not in self.unwanted['stat_lexicon']]

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

            if any(stem_word(i).lower() in self.unwanted['key_words_keep'] for i in split_text_wrds):
                if re.search('(levels|level)$', text_wrds):
                    cleaned_labels_outcomes.append((label, text_wrds))
                elif re.search('(NN.{0,1}\sIN)', text_pos):
                    for i,j in zip(text_wrds.split(','), text_pos.split(',')):
                        if re.search('^( CC|V.{0,2})', j):
                            cleaned_labels_outcomes.append((label, ' '.join(i for i in i.split()[1:])))
                        elif re.search('time', i):
                            cleaned_labels_outcomes.append((label, ' '.join(i for i in i.split()[1:])))
                        else:
                            cleaned_labels_outcomes.append((label, i))
                elif any(stem_word(i) in self.unwanted['key_words_keep'] for i in text_wrds.lower().split() if i.__contains__('scale')):
                    text_wrds = re.sub('(scale|scales)', '', text_wrds, flags=re.IGNORECASE)
                    cleaned_labels_outcomes.append((label, i))
                else:
                    text_wrds = re.sub('(\sand\s)|(\sor\s)|(\s;\s)',',', text_wrds)
                    for v in re.split(',+', text_wrds):
                        cleaned_labels_outcomes.append((label,i))


            # if  label != 'Mental':
            #     if any(stem_word(i.lower()) in self.unwanted['key_words_keep'] for i in split_text_wrds):
            #         cleaned_labels_outcomes.append((label,''))

            #scenario one: ALL phrases are nouns
            elif (all(i.__contains__('NN') for i in split_text_pos)):
                cleaned_labels_outcomes.append((label, text_wrds))

            #scenario 2: only one conjunction
            elif all(i.__contains__('NN') or i == 'CC' for i in split_text_pos):
                if(len([j for j in split_text_pos if j == 'CC']) == 1):
                    for x in re.split('and |or |; |,', text_wrds):
                        cleaned_labels_outcomes.append((label, x))
                elif(len([j for j in split_text_pos if j == 'CC']) > 1):
                    cleaned_labels_outcomes.append((label, ' '.join(text_wrds.split()[-2:])))

            # scenario 3: only one conjunction and one injunction
            elif all(i.__contains__('NN') or i == 'IN' or i == 'CC' for i in split_text_pos):
                if (len([j for j in split_text_pos if j == 'CC']) == 1):
                    x = re.search(r'^(NN.{0,2})+\sIN\s(NN.{0,2})\sCC\s(NN.{0,2})', text_pos)
                    x1 = re.search(r'^(NN.{0,2})+\sCC\s.+', text_pos)
                    if x is not None:
                        if x.group()[0:6] == 'NNS IN' and stem_word(text_wrds.split()[0]):
                            for x in re.split('and|or|;', text_wrds):
                                cleaned_labels_outcomes.append((label, x))
                        else:
                            cleaned_labels_outcomes.append((label, text_wrds))
                    elif x1 is not None:
                        if all(i == 'NN' or i == 'IN' or i == 'CC' for i in split_text_pos):
                            cleaned_labels_outcomes.append((label, text_wrds))
                        else:
                            sc4_split_index = split_text_pos.index('IN')
                            phrase_6 = ' '.join(i for i in split_text_wrds[(sc4_split_index+1):])
                            cleaned_labels_outcomes.append((label, phrase_6))
                    else:
                        t = re.compile('(IN\sCC\sIN\sNN)$')
                        if t.search(text_pos):
                            phrase_7 = ' '.join(i for i in split_text_wrds[:-4])
                            cleaned_labels_outcomes.append((label, phrase_7))
                        else:
                            cleaned_labels_outcomes.append((label, text_wrds))
                else:
                    cleaned_labels_outcomes.append((label, text_wrds))

            #scenario 4 and adjective and a conjunctions
            elif all(i.__contains__('NN') or i == 'JJ' or i == 'CC' for i in split_text_pos):
                if 'CC' in split_text_pos:
                    sc5_split_index = split_text_pos.index('CC')
                    if split_text_wrds[sc5_split_index-1] in split_text_wrds[sc5_split_index+1:]:
                        phrase_8 = ' '.join(i for i in split_text_wrds[:sc5_split_index])
                        phrase_9 = ' '.join(i for i in split_text_wrds[(sc5_split_index+1):])
                        cleaned_labels_outcomes.append((label, phrase_8))
                        cleaned_labels_outcomes.append((label, phrase_9))
                    else:
                        cleaned_labels_outcomes.append((label, text_wrds))
                else:
                    if not nltk.pos_tag([text_wrds])[0][1].__contains__('JJ'):
                        cleaned_labels_outcomes.append((label, text_wrds))

            elif all(i.__contains__('NN') or i == 'JJ' or i == 'CC' or i == '(' or i == ')' for i in split_text_pos):
                cleaned_labels_outcomes.append((label, text_wrds))
            elif all(i.__contains__('NN') or i == ',' for i in split_text_pos):
                for i in text_wrds.split(','):
                    cleaned_labels_outcomes.append((label, i))
            elif all(i.__contains__('NN') or i == ',' or i == 'CC' for i in split_text_pos):
                for i in re.split(', |and |or |; ', text_wrds):
                    if i:
                        cleaned_labels_outcomes.append((label, i))
            elif all(i.__contains__('NN') or i == ',' or i == 'IN' or i == 'JJ'  or i == 'CC' for i in split_text_pos):
                #text_wrds = re.sub(r'(\sand\s)|(\sor\s)', ',', text_wrds)
                for u in re.split(' +and +| +or +|, *|; *', text_wrds):
                    if u:
                        cleaned_labels_outcomes.append((label,i))
            elif all(i.__contains__('NN') or i.__contains__('V')  or i == 'CC' for i in split_text_pos):
                if re.search(r'V.{1,2}$', text_pos):
                    for u in re.split(' +and +| +or +|, *|; *', text_wrds):
                        cleaned_labels_outcomes.append((label,u))
                else:
                    for x,y in zip(split_text_wrds, split_text_pos):
                        if y.__contains__('V') or y.__contains__('CC') or y.__contains__('IN'):
                            split_text_wrds = split_text_wrds[1:]
                        else:
                            break
                    cleaned_labels_outcomes.append((label, ' '.join(i for i in split_text_wrds)))

            elif all(i.__contains__('NN') or i.__contains__('V') or i == 'IN'  or i == 'CC' for i in split_text_pos):
                for u in re.split(' +and +| +or +|, *|; *', text_wrds):
                    cleaned_labels_outcomes.append((label, i))
            elif all(i.__contains__('NN') or i.__contains__('V') or i == 'IN' or i == 'CC' or i == 'JJ' for i in split_text_pos):
                if re.search(r'(CC\sJJ\sNN\sIN\sNN)$', text_pos):
                    cleaned_labels_outcomes.append((label, text_wrds))
                else:
                    for u in re.split(' +and +| +or +|, *|; *', text_wrds):
                        cleaned_labels_outcomes.append((label, u))
            elif any(i == '(' or i == ')' or i == '[' or i == ']' for i in split_text_pos):
                for h in re.finditer(r'[\[\(]\s+[\w\s,;:%/._+-]+\s+[\)\]]', text_wrds):
                    is_english =  [check_english(x) for x in h.group().split()[1:-1]]
                    if len([i for i in is_english if i == True]) < len([i for i in is_english if i == False]):
                        text_wrds = re.sub('[\[\(]\s+[\w\s,;:%/._+-]+\s+[\)\]]','', text_wrds)
                text_wrds = (re.sub('(\(|\[|\)|\])', ',', text_wrds))
                text_pos = (re.sub('(\(|\[|\)|\])', ',', text_pos))
                for p, q in zip([w.strip() for w in text_wrds.split(',')], [h.strip() for h in text_pos.split(',')]):
                    if (re.search('^CC', q)):
                        cleaned_labels_outcomes.append((label, ' '.join(i for i in p.strip()[1:])))
                    elif q == 'DT' or q == 'CD' or q == 'JJ' or len(p) < 2:
                        pass
                    elif len([i for i in q.split() if i.__contains__('J')]) > len([i for i in q.split() if i.__contains__('N')]):
                        cleaned_labels_outcomes.append((label, p))
                    elif re.search('^NN\sIN', q):
                        to_add = (' '.join(i for i in p.split()[2:]))
                        if to_add:
                            cleaned_labels_outcomes.append((label, to_add))
                    else:
                        cleaned_labels_outcomes.append((label, p))
            elif re.search('^(NN.{0,1})+\sIN', text_pos):
                text_wrds = ' '.join(i for i in text_wrds.split()[2:])
                if re.search('IN\sCC\sIN\s(DT)?\sNN$', text_pos):
                    cleaned_labels_outcomes.append((label, ' '.join(i for i in text_wrds.split()[:-5])))
                for i in text_wrds.split(','):
                    cleaned_labels_outcomes.append((label, i))
            elif any(i == ',' for i in split_text_wrds):
                for u,v in zip(re.split(',+| and ', text_wrds), re.split(',+| CC ', text_pos)):
                    if v.__contains__('V'):
                        for i in re.split('was|measured| by', u):
                            if i:
                                cleaned_labels_outcomes.append((label,i))
                    else:
                        cleaned_labels_outcomes.append(((label, u)))
            else:
                if text_pos.__contains__('FW'):
                    cleaned_labels_outcomes.append((label, text_wrds))
                elif text_pos.__contains__('V'):
                    for u in (split_text_pos):
                        if split_text_pos[0].__contains__('V') or split_text_pos[0].__contains__('R'):
                            split_text_wrds = split_text_wrds[1:]
                    cleaned_labels_outcomes.append((label, ' '.join(i for i in split_text_wrds)))
                elif re.search('IN\sDT\sNN\sIN', text_pos):
                    for u in re.split(' with a reduction in ', text_wrds):
                        cleaned_labels_outcomes.append((label, u))
                elif re.search('(CD\sCC\sNN)$', text_pos):
                    cleaned_labels_outcomes.append((label, text_wrds))
                else:
                    for u in re.split('and', text_wrds):
                        cleaned_labels_outcomes.append((label, u))


        cleaned_labels_outcomes_dict = dict(cleaned_labels_outcomes)
        df_frame = pd.DataFrame(cleaned_labels_outcomes)
        df_frame.columns = ['Label','Outcome']
        print(df_frame)

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

def extracting_statistical_lexicon():
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
                            if d_3.lower() not in ['toxicity']:
                                statistical_terms.append(d_3.lower())
    return statistical_terms

def visulize_statistical_term_occurrence(x):
    outcomes_str = ' '.join([i for i in x])
    outcomes_str = [i for i in outcomes_str.split() if i.lower() in extracting_statistical_lexicon()]
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