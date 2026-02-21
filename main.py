import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

data = pd.read_csv("house_data.csv")

X = data[["area", "bedrooms", "age", "location_rating"]]
y = data["price"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

score = model.score(X_test, y_test)
print("Model R^2 Score:", score)

area = float(input("Enter area (sq ft): "))
bedrooms = int(input("Enter number of bedrooms: "))
age = float(input("Enter age of house: "))
location = float(input("Enter location rating (1-10): "))

prediction = model.predict([[area, bedrooms, age, location]])
print("Estimated House Price:", int(prediction[0]))
