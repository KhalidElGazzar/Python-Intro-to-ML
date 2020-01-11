import pandas as pd

# path of file to read
melb_file_path = './input/melbourne_data/melb_data.csv'

# read the data and store it in a DataFrame
melb_data = pd.read_csv(melb_file_path)

print("Loading Melbourne Data ..")

# print columns
print(melb_data.columns)

# print summary of the data
print(melb_data.describe())

print("Data loaded ..")
