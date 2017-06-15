import pandas as pd

shift = 4
sj_train = pd.read_csv('sj_train_original.csv')
iq_train = pd.read_csv('iq_train_original.csv')
sj_test = pd.read_csv('sj_test_original.csv')
iq_test = pd.read_csv('iq_train_original.csv')
pd_submit = pd.read_csv('dataset/submission_format.csv')

total_cases_sj = list(sj_train['total_cases'])
total_cases_iq = list(iq_train['total_cases'])

# Shift response column up
total_cases_sj = total_cases_sj[shift:]
total_cases_iq = total_cases_iq[shift:]

# Adjust training data to match response column
sj_train = sj_train[:-shift]
iq_train = iq_train[:-shift]

# Combine training columns with response column
sj_train['total_cases'] = total_cases_sj
iq_train['total_cases'] = total_cases_iq
sj_train.to_csv("iq_shift_train.csv", index=False)
