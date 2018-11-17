import sys
import pandas as pd
sys.path.append('/users/phd/micheala/Github/EBM-NLP-master')
from nltk.tokenize import word_tokenize
import ebm_nlp_demo as e
from pathlib import Path
import re
import string

class annotate_text:

    def __init__(self):
        self.turker, self.ebm_extract = e.read_anns('hierarchical_labels', 'outcomes', \
                                       model_phase='test/gold')

    def xml_wrapper(self):
        new_f = open('train_spans_labels.txt', 'w')
        l,s = 'LABEL','TEXT SPAN'
        new_f.write('{} : {}'.format(l, s)+'\n')
        annotate_f = open('test_set.txt', 'w')
        error_file = open('error.txt', 'a')
        error_file.write('{} : {}'.format(l, s) + '\n')
        for pmid, doc in self.ebm_extract.items():
            searched_text = ' '.join(word_tokenize(str(doc.text)))
            lab_span = e.print_labeled_spans_2(doc)
            #print(searched_text+'\n')
            for tup in lab_span:
                if(tup[1] not in string.punctuation):
                    new_f.write('{} : {}'.format(str(tup[0]),str(tup[1]))+'\n')
                    try:
                        searched_text = re.sub(tup[1], '<Outcome {}> {} </Outcome>'.format(tup[0], tup[1]), searched_text)
                    except Exception as d:
                        error_file.write('{} : {}'.format(tup[0],tup[1]) + '\n')
                        print (d, tup[0], tup[1])
            annotate_f.write('{}'.format(pmid) + '\n' + '\n')
            annotate_f.write(searched_text + '\n' + '\n')
        new_f.close()
        annotate_f.close()


def sub_span(label,old_span,data):
    new_span  = re.sub(old_span,'<Outcome {}>{}</outcome>'.format(label, old_span), data)
    return new_span

run = annotate_text()
run.xml_wrapper()









