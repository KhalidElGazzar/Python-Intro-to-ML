import pandas as pd

print("Loading Training Data ..")

# path of file to read
iowa_file_path = './input/home_data/train.csv'

# read the data and store it in a DataFrame
iowa_data = pd.read_csv(iowa_file_path)

'''
if the data has some missing cells, then the simplist option for now is to drop
the houses with missing data from the dataset
We will use the dropna method where na means N/A
'''
# The iowa dataset (training data) is complete so there is no need to use dropna method
# iowa_data = iowa_data.dropna(axis=0)

# print columns
print(iowa_data.columns)

# print summary of the data
print(iowa_data.describe())

print("Data Loaded")