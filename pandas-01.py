import pandas as pd

# path of file to read
melb_file_path = './input/melbourne_data/melb_data.csv'

# read the data and store it in a DataFrame
melb_data = pd.read_csv(melb_file_path)

# print summary of the data
print(melb_data.describe())

