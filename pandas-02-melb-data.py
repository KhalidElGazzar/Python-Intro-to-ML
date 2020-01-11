import pandas as pd

print("Loading Melbourne Data ..")

# path of file to read
melb_file_path = './input/melbourne_data/melb_data.csv'

# read the data and store it in a DataFrame
melb_data = pd.read_csv(melb_file_path)

'''
if the data has some missing cells, then the simplist option for now is to drop
the houses with missing data from the dataset
We will use the dropna method where na means N/A
'''
melb_data = melb_data.dropna(axis=0)

# print columns
print(melb_data.columns)

# print summary of the data
print(melb_data.describe())

print("Data loaded ..")
