import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib as jb


def load_data():
    # Your dataset loading logic here. Load the California housing dataset and convert it to a pandas DataFrame
    housing = fetch_california_housing()
    housing_df = pd.DataFrame(data=housing.data, columns=housing.feature_names)
    housing_df['MedianHouseValue'] = housing.target
    return housing_df

def main():

    # Load the dataset   
    housing_df = load_data()
    
    # Split the dataset into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(housing_df.iloc[:, :-1], housing_df.iloc[:, -1], test_size=0.2, random_state=42)

    # Train a linear regression model on the training data
    model = LinearRegression()
    model.fit(x_train, y_train)

    # Make predictions on the test data
    y_pred = model.predict(x_test)

    # Evaluate the model's performance using the R-squared score
    model_score = model.score(x_test, y_test)

    # Save the trained model
    jb.dump(model, 'reg_model.joblib')


    # Print the first few rows of the dataset and the shapes of the training and testing sets, as well as the model's R-squared score
    print(housing_df.head())
    print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
    print("Model R-squared score:", model_score)




if __name__ == "__main__":
    main()
