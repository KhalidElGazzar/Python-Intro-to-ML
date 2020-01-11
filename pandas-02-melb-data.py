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

print("Selecting Data for modeling")

print ("Set the prediction target y (Price)")
# select the prediction target:
# usually called y by convention
# it is a single column stored in Series. A Series is a strcuture like DataFrame but with 
# single column of data
y = melb_data.Price

print("Set the Features X: 'Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude' ")
# select the Features - called X by convention
# Features are the columns that are inputted in the model and will be used to determine
# the prediction target y (price in our case)
# Sometimes we use all columns except the prediction target as features.
melb_features = ['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']
X = melb_data[melb_features]

# describe the Features
print(X.describe())

# show the first few (10) rows of the data
# print(X.head(10))
print(X.head(10))

