import pandas as pd
import tensorflow as tf
#import thundersvm as svm
from sklearn import svm
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, classification_report, confusion_matrix
import numpy as np
import warnings
from tabulate import tabulate
import sys
import time
import pickle
from base_models import data_prep

sys.path.append('/users/micheala/pico-outcome-prediction/')

warnings.filterwarnings("ignore", category=Warning)

# defining parameters
tf.flags.DEFINE_float("sample_percentage", .2, "Percentage of data to be used for validation")
tf.flags.DEFINE_string("ebm_file", "../labels_outcomes_2.csv", "Data source.")

FLAGS = tf.flags.FLAGS


def classification_report_and_accuracy(y_true, y_pred, target_classes):
    accuracy = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=target_classes)
    conf_matrix = confusion_matrix(y_true, y_pred)
    p_r_f_s = precision_recall_fscore_support(y_true, y_pred, labels=target_classes)
    return accuracy, report, conf_matrix


def data_split(test_percentage):
    data = pd.read_csv(FLAGS.ebm_file)
    class_dict = data_prep.get_number_in_classes(data['Label'])
    org_count_perclass = ['{}:{:d}'.format(x, y) for x, y in class_dict.items()]

    # randomly shuffle the dataset
    indices = np.random.permutation(data['Outcome'].shape[0])
    shuffled_x = data['Outcome'][indices]
    shuffled_y = data['Label'][indices]

    # obtain training and testing sets
    shuffled_xtdf, voc_tdf, shuffled_xct, voc_ct = data_prep.n_grams(shuffled_x)
    X_train, X_test, y_train, y_test = train_test_split(shuffled_xtdf, shuffled_y, test_size=test_percentage)
    train_set = ['{}:{:d}'.format(x, y) for x, y in data_prep.get_number_in_classes(y_train).items()]
    test_set = ['{}:{:d}'.format(x, y) for x, y in data_prep.get_number_in_classes(y_test).items()]
    print('Original dataset \n{}\nTraining \n{}\nTest \n{}'.format(org_count_perclass, train_set, test_set))

    return X_train, X_test, y_train, y_test, class_dict


def svm_classifier(xtrain, ytrain, classes):
    entries = []
    svm_model = svm.SVC(kernel='linear', C=0.0001, gamma=0.05)
    svm_model_frame = pd.DataFrame()

    skf = StratifiedKFold(n_splits=5, random_state=None, shuffle=True)

    print('--------------Training-Accuracy --------------\n')
    s = open('svm-original.txt', 'w')
    fold = 1
    try:
        with open('svm.txt', 'w') as sv:
            for train_index, test_index in skf.split(xtrain, ytrain):
                x_train, y_train = xtrain[train_index], ytrain[train_index]
                x_test, y_test = xtrain[test_index], ytrain[test_index]
                start_time = time.time()
                vsm = svm_model.fit(x_train, y_train)
                end_time = time.time()
                y_pred = vsm.predict(x_test)
                tr_predict = svm_model.predict(x_train)
                print(end_time - start_time)
                entries.append((fold,
                                '{:.4f}'.format((end_time - start_time)),
                                '{:.4f}'.format(svm_model.score(x_train, y_train)),
                                '{:.4f}'.format(accuracy_score(y_test, y_pred))))

                s.write('fold:{}\nValidation Accuracy: {}\nTraining Accuracy: {}\nValidation Metrics \n {} \n Training Metrics \n {}'.format(
                        fold, accuracy_score(y_test, y_pred) * 100, vsm.score(x_train, y_train) * 100,
                        classification_report_and_accuracy(y_test, y_pred, classes),
                        classification_report(y_train, tr_predict, classes)))
                fold += 1

            svm_model_frame = pd.DataFrame(entries, columns=['Fold', 'Time(sec)', 'Train Accuracy', 'Test Accuracy']).sort_values(by='Test Accuracy', ascending=False)
            print(tabulate(svm_model_frame, headers='keys', tablefmt='psql'))

            s = pd.DataFrame(entries, columns=['Fold', 'Time(sec)', 'Train Accuracy', 'Test Accuracy']).sort_values(by='Test Accuracy', ascending=False)
            s = s.round(4)

            sv.write('Mean Time per fold:{:f} Mean train Accuracy score:{:f} Mean Validation Accuracy score:{:f}'.format(
                    np.mean(svm_model_frame['Time(sec)'].astype(float)),
                    np.mean(svm_model_frame['Train Accuracy'].astype(float)) * 100,
                    np.mean(svm_model_frame['Test Accuracy'].astype(float)) * 100))

    except Exception as e:
        print(e)

    print('--------------END --------------\n')
    filename = 'svm_model.sav'
    pickle.dump(svm_model, open(filename, 'wb'))
    return svm_model


def evaluate_model(model, xtest, ytest, classes):
    y_pred = model.predict(xtest)
    test_set = ['{}:{:d}'.format(x,y) for x,y in data_prep.get_number_in_classes(ytest).items()]
    print('test-set\n{}'.format(test_set))
    print('--------------Testing Accuracy--------------\n')
    for i in classification_report_and_accuracy(ytest, y_pred, classes):
        print('{}\n'.format(i))
    print('--------------END---------------\n')


if __name__ == '__main__':
    X_train, X_test, y_train, y_test, class_dict = data_split(FLAGS.sample_percentage)

    saved_model = svm_classifier(X_train[:500], y_train.values[:500], list(class_dict))

    #load saved model
    start_test_time = time.time()
    evaluate_model(saved_model, X_test, y_test, list(class_dict))
    end_test_time = time.time()
    print('Testing-time:{:.4f}'.format(end_test_time-start_test_time))

