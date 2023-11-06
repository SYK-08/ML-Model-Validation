import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# importing training data 
iowa_file_path = "D:/Data/train_kaggle.csv"

# converting csv to dataframe
home_data = pd.read_csv(iowa_file_path)
# print(home_data.columns)

# specifying prediction target (SalePrice is a column in home_data)
y = home_data.SalePrice

# creating a variable that holds data that are relevant for predicting prices
feature_names = ["LotArea", "YearBuilt", "1stFlrSF", "2ndFlrSF", "FullBath", "BedroomAbvGr", "TotRmsAbvGrd"]

#creating a subset of dataframe home_data containing the columns in "feature_names"
X = home_data[feature_names]

# specifying the model for training
iowa_model = DecisionTreeRegressor(random_state = 1)
iowa_model.fit(X, y)

# making predictions
predictions = iowa_model.predict(X)
print("Initial prediction: " ,predictions)

# splitting training data
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 1)

# fitting new data
iowa_model.fit(train_X, train_y)

# making predictions on validation data
val_predictions = iowa_model.predict(val_X)
print("Prediction after validation" ,val_predictions)

# calculating Mean Absolute Error in Validation Data
val_mae = mean_absolute_error(val_y, val_predictions)
print("Mean Absolute Error:", val_mae)
