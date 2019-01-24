import os
import pandas as pd

from nltk.tokenize import word_tokenize
import ebm_nlp_demo as e
import re
from glob import glob

class annotate_text:

    def __init__(self):
        self.turker, self.ebm_extract = e.read_anns('hierarchical_labels', 'outcomes', \
                                      ann_type = 'aggregated', model_phase = 'train')

    def xml_wrapper(self):

        main_dir = 'adding_tags_to_ebm/'
        data_dir = os.path.abspath(os.path.join(main_dir, 'aggregated', 'train_spans.txt'))
        if not os.path.exists(os.path.dirname(data_dir)):
            os.makedirs(os.path.dirname(data_dir))
        print('Tagged spans are being written in file location %s'%(data_dir))
        lab_outcome = []
        lab_outcome.clear()
        context_file = open('lack_context.txt', 'w')

        with open(data_dir, 'w') as new_f:
            for pmid, doc in self.ebm_extract.items():
                text = (doc.text).lower()
                outcomes = list(set(e.print_labeled_spans_2(doc)))
                outcomes_ordered_by_len = sorted(outcomes, key=lambda x: len(x[1]), reverse=True)

                #print('\n'+text+'\n',outcomes_ordered_by_len)
                mental_outcomes = [i for i in list(set(outcomes_ordered_by_len)) if i[0] == 'Physical' and str(i[1]).__contains__('carcass performance')]
                for elem in list(set(mental_outcomes)):
                    # phrase = re.sub('[.]$','', elem[1])
                    # #text = text.replace(phrase, '<Outcome {}>{}</Outcome>'.format(elem[0], phrase.upper()))
                    #
                    # lab_outcome.append(elem)
                    # if str(elem[0]) == 'Mental':
                    #     if(str(elem[1]).__contains__('number of picture exchanges in a far-transfer')):
                    #         context_file.write(text)
                    #     else:
                    context_file.write(text+'\n')
                    break





                #new_f.write('PMID - file %s\n'%(pmid)+text)
            context_file.close()
        return set(outcomes_ordered_by_len)

    def df_frame(self):
        discovered_outcomes =  list(self.xml_wrapper())
        discovered_outcomes_organised = sorted(discovered_outcomes, key=lambda l : l[0])
        cleaned_labels_outcomes = []
        for tup in discovered_outcomes_organised:
            '''look for words inside circle or square braces within the e.g. "Lung injuries [ 'TPN' ]" contains a word TPN in square braces and if found, 
            strip off the spaces and retain a cleaned outcome'''
            pattern_x = re.compile(r'[\[\(]\s+[\w\s,;:%/._+-]+\s+[\)\]]')
            irregular_string = pattern_x.findall(tup[1])
            if irregular_string:
                out_come = str(tup[1])
                for item in irregular_string:
                    regularised_string = item[0] + item[2:-2] + item[-1]
                    out_come = out_come.replace(item, regularised_string)
            else:
                out_come = str(tup[1])
            cleaned_outcome = re.sub(r'[-_+.,;:]$', '', out_come)
            if cleaned_outcome:
                cleaned_labels_outcomes.append((tup[0], cleaned_outcome.strip()))

        cleaned_labels_outcomes_dict = dict(cleaned_labels_outcomes)
        df_frame = pd.DataFrame.from_dict(cleaned_labels_outcomes)
        df_frame.columns = ['Label','Outcome']

        new_dir = os.path.abspath('adding_tags_to_ebm/aggregated')
        if not os.path.exists(new_dir):
            os.makedirs(new_dir)
        df_frame.to_csv(os.path.join(new_dir, 'aggregated_test_crowd.csv'))

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
    for ann in ann_type:
        locate_dir = os.path.abspath(os.path.join('adding_tags_to_ebm', ann))
        csv_files = glob(os.path.join(locate_dir, '*.csv'))

        for file in csv_files:
            f = pd.read_csv(file)
            f = f[['Label','Outcome']]
            list_of_frames.append(f)

    concat_frames = pd.concat(list_of_frames)
    concat_frames_sorted = concat_frames.sort_values(by='Label')
    print(concat_frames_sorted)

    concat_frames_sorted.drop_duplicates(subset=['Label','Outcome'], keep=False)
    concat_frames_sorted['Outcome'] = concat_frames_sorted['Outcome'].apply(lambda x: sub_span(x))
    concat_frames_sorted = concat_frames_sorted.loc[concat_frames_sorted['Outcome'].str.len() > 1]
    print(concat_frames_sorted.shape)


    concat_frames_sorted.to_csv('labels_outcomes.csv')



#final_label_outcome(['aggregated', 'individual'])

run = annotate_text()
run.xml_wrapper()