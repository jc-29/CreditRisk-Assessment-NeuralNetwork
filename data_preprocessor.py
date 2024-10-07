import numpy as np
import scipy.stats as stats

FILE = open('credit_risk_dataset.csv', 'r')
DATA = np.genfromtxt(FILE, delimiter=',', dtype=None, names=True, encoding=None)

def convert_str_to_int():
    '''
    Assign an integer value to each string in the dataset.
    '''

    # corresponding integers for each string in the person_home_ownership column
    person_home_ownership = {
        'MORTGAGE': 0,
        'RENT': 1,
        'OWN': 2,
        'OTHER': 3
    }

    # corresponding integers for each string in the loan_intent column
    loan_intent = {
        'HOMEIMPROVEMENT': 0,
        'MEDICAL': 1,
        'PERSONAL': 2,
        'VENTURE': 3,
        'DEBTCONSOLIDATION': 4,
        'EDUCATION': 5
    }

    # corresponding integers for each string in the loan_grade column
    loan_grade = {
        'A': 0,
        'B': 1,
        'C': 2,
        'D': 3,
        'E': 4,
        'F': 5,
        'G': 6
    }

    # corresponding integers for each string in the cb_person_default_on_file column
    cb_person_default_on_file = {
        'Y': 0,
        'N': 1
    }

    # convert the strings to integers for each row in the dataset
    for i in range(len(DATA)):
        DATA[i]['person_home_ownership'] = person_home_ownership[DATA[i]['person_home_ownership']]
        DATA[i]['loan_intent'] = loan_intent[DATA[i]['loan_intent']]
        DATA[i]['loan_grade'] = loan_grade[DATA[i]['loan_grade']]
        DATA[i]['cb_person_default_on_file'] = cb_person_default_on_file[DATA[i]['cb_person_default_on_file']]

def save_data(filename, data):
    '''
    Save the preprocessed data to a new csv file.
    '''
    np.savetxt(filename, data, delimiter=',', fmt='%s', header=','.join(DATA.dtype.names), comments='')

def run(): 
    convert_str_to_int()

    mean = format(np.nanmean(DATA['loan_int_rate']), '.2f')
    median = format(np.nanmedian(DATA['loan_int_rate']), '.2f')
    mode = format(stats.mode(DATA['loan_int_rate'], nan_policy='omit').mode[0], '.2f')
    remove_indices = []

    # save preprocessed data with nan values replaced by mean
    data_mean = DATA.copy()
    for i in range(len(data_mean)):
        if np.isnan(data_mean[i]['loan_int_rate']):
            remove_indices.append(i) # to be used for later
            data_mean[i]['loan_int_rate'] = mean
    save_data('credit_risk_dataset_preprocessed_mean.csv', data_mean)

    # save preprocessed data with nan values replaced by median
    data_median = DATA.copy()
    for i in range(len(data_median)):
        if np.isnan(data_median[i]['loan_int_rate']):
            data_median[i]['loan_int_rate'] = median
    save_data('credit_risk_dataset_preprocessed_median.csv', data_median)    
    
    # save preprocessed data with nan values replaced by mode
    data_mode = DATA.copy()
    for i in range(len(data_mode)):
        if np.isnan(data_mode[i]['loan_int_rate']):
            data_mode[i]['loan_int_rate'] = mode
    save_data('credit_risk_dataset_preprocessed_mode.csv', data_mode)

    # save preprocessed data with rows with nan values removed
    data_uninclude = DATA.copy()
    i = 0
    while i < len(data_uninclude):
        if np.isnan(data_uninclude[i]['loan_int_rate']):
            data_uninclude = np.delete(data_uninclude, i)
        else:
            i += 1
    save_data('credit_risk_dataset_preprocessed_uninclude.csv', data_uninclude)

    # save preprocessed data with nan values left in and unchanged
    save_data('credit_risk_dataset_preprocessed.csv', DATA)

if __name__ == '__main__':
    run()

