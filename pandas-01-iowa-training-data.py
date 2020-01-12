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

print("Selecting Data for modeling")

print ("Set the prediction target y (SalePrice)")
# select the prediction target:
# usually called y by convention
# it is a single column stored in Series. A Series is a strcuture like DataFrame but with 
# single column of data
y = iowa_data.SalePrice

print("Set the Features X: 'Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude' ")
# select the Features - called X by convention
# Features are the columns that are inputted in the model and will be used to determine
# the prediction target y (price in our case)
# Sometimes we use all columns except the prediction target as features.
iowa_features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
X = iowa_data[iowa_features]

# describe the Features
print(X.describe())

# show the first few (10) rows of the data
# print(X.head(10))
print(X.head())


# Now, lets build the model
'''
We will use Scikit-learn library to create our model. This library is written as sklearn in our code
It is easily the most popular library for modeling the types of data typically stores in DataFrames

STEPS IN BUILDING & USING A MODEL
1. Define:
    - model type (Ex: DT or other type)
    - model params
2. Fit:
    Capture patterns from provided data (This is the heart of modelling)
3. Predict:
    Crystal clear
4. Evaluate:
    Measure the accuracy of your model
'''

from sklearn.tree import DecisionTreeRegressor

# define model: To ensure having same results each run, use random_state=1
iowa_model = DecisionTreeRegressor(random_state=1)

# fit model
iowa_model.fit(X,y)

print("Making predictions for the following 5 houses:")
print(X.head())
print("The predictions are")
print(iowa_model.predict(X.head()))

print("The actual prices are")
print(iowa_data.SalePrice.head())
