# write your code here
import pandas as pd

# import numpy as np
import matplotlib.pyplot as plt

# pd.set_option('display.max_columns', 8)

general = pd.read_csv(
    r'/Users/msorokina/PycharmProjects/Data Analysis for Hospitals/Data Analysis for Hospitals/task/test/general.csv')
prenatal = pd.read_csv(
    r'/Users/msorokina/PycharmProjects/Data Analysis for Hospitals/Data Analysis for Hospitals/task/test/prenatal.csv')
sports = pd.read_csv(
    r'/Users/msorokina/PycharmProjects/Data Analysis for Hospitals/Data Analysis for Hospitals/task/test/sports.csv')

# 2.1 Change the column names.
# All column names in the sports and prenatal tables must match the column names in the general table
gen_columns = general.columns.tolist()
# print(gen_columns)
# prenatal = prenatal.rename(columns={'HOSPITAL': 'hospital', 'Sex': 'gender'})
# sports = sports.rename(columns={'Hospital': 'hospital', r'Male/female': 'gender'})
prenatal.columns = general.columns
sports.columns = general.columns

# 2.2 Merge the DataFrames into one.
full_hospital = pd.concat([general, prenatal, sports], ignore_index=True)

# 2.3 Delete the Unnamed: 0 column
full_hospital = full_hospital.drop(full_hospital.columns[0], axis=1)

# 2.4 Print random 20 rows of the resulting DataFrame.
sample_of_hospital = full_hospital.sample(n=20, random_state=30)
# print(sample_of_hospital)

# 3.1 Delete all the empty rows
# mask = full_hospital.isnull()
# rows_mask = mask.any(axis=1)
full_hospital = full_hospital.dropna(axis=0, how='all')

# 3.2 Correct all the gender column values to f and m respectively
dict_1 = {'female': 'f', 'woman': 'f', 'male': 'm', 'man': 'm'}
full_hospital['gender'] = full_hospital['gender'].replace(dict_1)

# 3.3 Replace the NaN values in the gender column of the prenatal hospital with f
full_hospital.loc[full_hospital['hospital'] == 'prenatal', 'gender'] = full_hospital.loc[
    full_hospital['hospital'] == 'prenatal', 'gender'].fillna('f')

# 3.4 Replace the NaN values in the bmi, diagnosis, blood_test, ecg, ultrasound, mri, xray, children, months
# columns with zeros
columns_with_nan = ['bmi', 'diagnosis', 'blood_test', 'ecg', 'ultrasound', 'mri', 'xray', 'children', 'months']
full_hospital[columns_with_nan] = full_hospital[columns_with_nan].fillna('0')
# print(f'Data shape: {full_hospital.shape}')
# print(full_hospital.sample(n=20, random_state=30))
#__________________________________________________________________________________
# 4.1 Which hospital has the highest number of patients?
# number_of_patients = full_hospital.groupby('hospital').count().idxmax()[0]
# print(f'The answer to the 1st question is {number_of_patients}')

# 4.2 What share of the patients in the general hospital suffers from stomach-related issues?
# Round the result to the third decimal place.
# only_general = full_hospital[full_hospital['hospital'] == 'general']
# diagnosis_general = only_general['diagnosis'].value_counts()
# total_general = full_hospital['hospital'].value_counts()
# number_of_stomach_issues = round(diagnosis_general['stomach'] / total_general['general'], 3)
# print(f'The answer to the 2nd question is {number_of_stomach_issues}')

# 4.3 What share of the patients in the sports hospital suffers from dislocation-related issues?
# Round the result to the third decimal place.
# number_of_sports_dislocation = round(
#    full_hospital.loc[full_hospital['hospital'] == 'sports'].loc[
#        full_hospital['diagnosis'] == 'dislocation'].count()[0] /
#    full_hospital.loc[full_hospital['hospital'] == 'sports'].count()[0], 3)
# print(f'The answer to the 3rd question is {number_of_sports_dislocation}')

# 4.4 What is the difference in the median ages of the patients in the general and sports hospitals?
# general_df = full_hospital[full_hospital['hospital'] == 'general']
# sports_df = full_hospital[full_hospital['hospital'] == 'sports']

# Calculate the median age for each hospital category
# general_median_age = general_df['age'].median()
# sports_median_age = sports_df['age'].median()

# Find the difference in the median ages
# age_difference = general_median_age - sports_median_age
# print(f'The answer to the 4th question is {age_difference}')

# 4.5 In which hospital the blood test was taken the most often
# (there is the biggest number of t in the blood_test column among all the hospitals)?
# How many blood tests were taken?
# blood_test_count = full_hospital.groupby('hospital')["blood_test"].value_counts()

#blood_test_amount = pd.pivot_table(full_hospital, values='blood_test', columns='hospital', aggfunc='count').loc[
 #   'blood_test', blood_test_place]
# print(f'The answer to the 5th question is {blood_test_count.idxmax()[0]}, {blood_test_count.max()} blood tests')

bins = [0, 15, 35, 55, 70, 80]

fig, ax = plt.subplots()

hist = ax.hist(full_hospital['age'], bins=bins, linewidth=0.5, edgecolor="white")

for i in range(len(bins) - 1):
    bin_start = bins[i]
    bin_end = bins[i + 1]
    bin_mid = (bin_start + bin_end) / 2
    bin_count = hist[0][i]

    # Подпись аннотации
    annotation_text = f"{bin_count}"

    # Добавление аннотации на график
    ax.annotate(annotation_text, (bin_mid, bin_count), xytext=(0, 5), textcoords='offset points', ha='center')
    # Установка подписей для оси X
    xtick_labels = ['0-15', '15-35', '35-55', '55-670', '70-80']
    xtick_positions = [(bins[i] + bins[i + 1]) / 2 for i in range(len(bins) - 1)]
    plt.xticks(xtick_positions, xtick_labels)
plt.show()
print(f'The answer to the 1st question: 15-35')

diagnosis_counts = full_hospital['diagnosis'].value_counts()
diagnosis_pie = diagnosis_counts.plot.pie(autopct='%1.1f%%', legend=False)
plt.show()
print(f'The answer to the 2nd question: pregnancy')

data_list = full_hospital['height']
fig, axes = plt.subplots()
plt.violinplot(data_list)
plt.show()

print(f'The answer to the 3rd question: the reason is that most of the patients are newborns.')

