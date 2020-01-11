import pandas as pd

# path of file to read
iowa_file_path = './input/home_data/train.csv'

# read the data and store it in a DataFrame
iowa_data = pd.read_csv(iowa_file_path)

print("Loading Training Data ..")

# print summary of the data
print(iowa_data.describe())

print("Data Loaded")