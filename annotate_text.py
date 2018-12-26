import os
import pandas as pd

from nltk.tokenize import word_tokenize
import ebm_nlp_demo as e
from pathlib import Path
import re
import string
import operator

class annotate_text:

    def __init__(self):
        self.turker, self.ebm_extract = e.read_anns('hierarchical_labels', 'interventions', \
                                      ann_type = 'aggregated', model_phase = 'test/gold')

    def xml_wrapper(self):

        main_dir = 'adding_tags_to_ebm/'
        data_dir = os.path.abspath(os.path.join(main_dir, 'aggregated', 'test_spans.txt'))
        if not os.path.exists(os.path.dirname(data_dir)):
            os.makedirs(os.path.dirname(data_dir))
        print('Tagged spans are being written in file location %s'%(data_dir))
        out_label, lab_outcome = {}, []
        lab_outcome.clear()
        with open(data_dir, 'w') as new_f:
            for pmid, doc in self.ebm_extract.items():
                text = (doc.text).lower()
                outcomes = list(set(e.print_labeled_spans_2(doc)))
                outcomes_ordered_by_len = sorted(outcomes, key=lambda x: len(x[1]), reverse=True)

                for elem in outcomes_ordered_by_len:
                    phrase = re.sub('[.]$','', elem[1])
                    text = text.replace(phrase, '<Outcome {}>{}</Outcome>'.format(elem[0], phrase.upper()))

                    lab_outcome.append(elem)

                new_f.write('PMID - file %s\n'%(pmid)+text)

        return set(lab_outcome)

    def df_frame(self):
        discovered_outcomes =  list(self.xml_wrapper())
        discovered_outcomes_organised = sorted(discovered_outcomes, key=lambda l : l[0])
        #discovered_outcomes_organised = [(i[0].strip(), i[0].strip().split)]
        cleaned_labels_outcomes = []
        for tup in discovered_outcomes_organised:
            pattern_x = re.compile(r'[\[\(]\s[\w\s/._+-]+\s[\)\]]')
            irregular_string = pattern_x.search(tup[1])
            if irregular_string:
                regularised_string = ''.join([char for char in irregular_string.group().split()])
                print(irregular_string.group(), '  ', regularised_string)
        #         out_come = re.sub(str(irregular_string.group()), str(regularised_string).strip('()'), tup[1])
        #     else:
        #         out_come = tup[1]
        #     cleaned_outcome = re.sub(r'[-_+.;:]$', '', out_come)
        #     cleaned_labels_outcomes.append((tup[0], cleaned_outcome.strip()))
        #
        # cleaned_labels_outcomes_dict = dict(cleaned_labels_outcomes)
        # df_frame = pd.DataFrame.from_dict(cleaned_labels_outcomes)
        # print(df_frame)
        # for i in cleaned_labels_outcomes:
        #     print(i)


        #     searched_text = ' '.join(word_tokenize(str(doc.text)))
        #     lab_span = e.print_labeled_spans_2(doc)
        #     #print(searched_text+'\n')
        #     for tup in lab_span:
        #         if(tup[1] not in string.punctuation):
        #             #new_f.write('{} : {}'.format(str(tup[0]),str(tup[1]))+'\n')
        #             try:
        #                 searched_text = re.sub(tup[1], '<Outcome {}> {} </Outcome>'.format(tup[0], tup[1]), searched_text)
        #             except Exception as d:
        #                 #error_file.write('{} : {}'.format(tup[0],tup[1]) + '\n')
        #                 print (d, tup[0], tup[1])
        #     #annotate_f.write('{}'.format(pmid) + '\n' + '\n')
        #     #annotate_f.write(searched_text + '\n' + '\n')
        # #new_f.close()
        # # annotate_f.close()



def sub_span(label, old_span, data):
    new_span  = re.sub(old_span, '<Outcome {}>{}</outcome>'.format(label, old_span.upper()), data)
    return new_span

run = annotate_text()
run.df_frame()