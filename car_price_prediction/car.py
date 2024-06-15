import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


df = pd.read_csv('C:/Users/hp/OneDrive/Documents/cardekho.csv')

df.dropna(subset=['selling_price'], inplace=True)

df['year'] = df['year'].astype(int)
df['km_driven'] = df['km_driven'].astype(int)

X = df.drop(columns=['name'])
y = df['selling_price']

numerical_cols = ['year', 'km_driven', 'mileage(km/ltr/kg)', 'engine']
categorical_cols = ['fuel', 'seller_type', 'transmission', 'owner']

numerical_transformer = Pipeline(steps=[
('imputer', SimpleImputer(strategy='median')),
('scaler', StandardScaler())
])
categorical_transformer = Pipeline(steps=[
('imputer', SimpleImputer(strategy='most_frequent')),
('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
transformers=[
('num', numerical_transformer, numerical_cols),
('cat', categorical_transformer, categorical_cols)
])

model = RandomForestRegressor(n_estimators=100, random_state=0)

pipeline = Pipeline(steps=[('preprocessor', preprocessor),
('model', model)
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
# print(f"Mean Absolute Error (MAE): {mae}")
# print(f"Mean Squared Error (MSE): {mse}")
# print(f"Root Mean Squared Error (RMSE): {rmse}")
def get_car_details_by_name(car_name):
 car_details = df[df['name'].str.contains(car_name, case=False)]
 return car_details

def predict_car_info():
  print("Enter the name of the car to get its details:")
name = input("Name: ")
car_details = get_car_details_by_name(name)
if car_details.empty:
 print("Car not found in the dataset.")
else:
 print("Car Details:")
print(car_details)

