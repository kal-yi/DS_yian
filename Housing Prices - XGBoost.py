import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, QuantileTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import cross_val_score
from sklearn.compose import make_column_selector as selector
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from xgboost import XGBRegressor


class LabelEncoderExt(object):

    # Improvement of the LabelEncoder as it gives an error when comes across unseen data on X_test)

    def __init__(self, *args, **kwargs):
        """
        It differs from LabelEncoder by handling new classes and providing a value for it [Unknown]
        Unknown will be added in fit and transform will take care of new item. It gives unknown class id
        """
        self.label_encoder = LabelEncoder(*args, **kwargs)
        # self.classes_ = self.label_encoder.classes_

    def fit(self, data_list, y=None):
        """
        This will fit the encoder for all the unique values and introduce unknown value
        :param data_list: A list of string
        :return: self
        """
        self.label_encoder = self.label_encoder.fit(list(data_list) + ['Unknown'])
        self.classes_ = self.label_encoder.classes_

        return self

    def transform(self, data_list, y=None):
        """
        This will transform the data_list to id list where the new values get assigned to Unknown class
        :param data_list:
        :return:
        """
        new_data_list = list(data_list)
        for unique_item in np.unique(data_list):
            if unique_item not in self.label_encoder.classes_:
                new_data_list = ['Unknown' if x==unique_item else x for x in new_data_list]

        return self.label_encoder.transform(new_data_list)

def data_load(X_filepath, X_test_filepath):

    # Read data, remove rows with missing target, and return X, y, X_test

    X_full = pd.read_csv(X_filepath, index_col="Id")
    X_test = pd.read_csv(X_test_filepath, index_col="Id")
    X_full.dropna(axis=0, subset=["SalePrice"], inplace=True)
    y=X_full.SalePrice
    X=X_full.drop(["SalePrice"], axis=1)

    return X, y, X_test

def data_visualisation(X,y):

    # Function plots each feature against the labels

    for col in X.columns:
        plt.figure()
        sns.scatterplot(x=X[col], y=y)
        plt.show()

        if X[col].dtype in ["int64", "float64"]:
            plt.figure()
            sns.kdeplot(data=X[col], shade=True)
            plt.show()

    return

def missing_values(X, X_test):

    # Use this function to check the features for missing values and make changes

    # Return a table with the number of missing values for each feature
    missing_val_count_by_column = (X.isnull().sum())
    print("Number of missing values for each feature:\n",missing_val_count_by_column[missing_val_count_by_column > 0])

    # Return a list with the unique values of each feature - focus on the columns with the many missing
    unique_per_col = list(map(lambda col: X[col].nunique(dropna=False), X.columns))
    d = dict(zip(X.columns, unique_per_col))
    print("Number of unique values of each feature:\n",sorted(d.items(), key=lambda x: x[1]))

    # Value_counts - focus on the columns with high NaN number
    #print(X["LotFrontage"].value_counts(dropna=False))

    # Replace NaN with No for the following columns as they do not have the feature
    for col in ["Alley","FireplaceQu","PoolQC","Fence","MiscFeature"]:
        X[col].fillna(value="No", inplace=True)
        X_test[col].fillna(value="No", inplace=True)

    return X, X_test

def preprocessing(numerical_cols,categorical_cols):

    # Function defines the preprocessor

    # Preprocessing for numerical data
    numerical_transformer = Pipeline(steps=[
        ("imputer" , SimpleImputer(strategy="most_frequent")),
        ("scaler", QuantileTransformer(n_quantiles=500))
    ])

    # Preprocessing categorical data
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
    #    ("labelencoder", LabelEncoderExt()) # combine this with low candinality categorical_cols
    #])
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    # Put together preprocessing for numerical and categorical data
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_transformer, numerical_cols),
            ("cat", categorical_transformer, categorical_cols)
     #       ("cat", categorical_transformer,  selector(dtype_include="category")) # when using the "selector" the I can use the LabelEncoder
        ])

    return preprocessor

def model_and_clf(preprocessor, n_estimators=700, learning_rate=0.02): #default n_estimators, learning_rate - unless overwritten

    # Function defines and returns the classifier

    model = XGBRegressor(n_estimators=n_estimators, learning_rate=learning_rate, n_jobs=-1, random_state=0)
    clf = Pipeline(steps=[("preprocessor", preprocessor),
                          ("model", model)
                          ])

    return clf

def get_preds(preprocessor, X_train, X_valid, y_train, n_estimators, learning_rate): #X_Valid or X_test to return preds

    # Create classifier
    clf1=model_and_clf(preprocessor, n_estimators, learning_rate)

    # Preprocessing of train data and fit to model
    clf1.fit(X_train, y_train)

    # Preprocessing of valid data and get predictions
    preds = clf1.predict(X_valid)

    return preds

def cross_validation(clf, X, y):
    scores = -1 * cross_val_score(clf, X, y,
                                  cv=5,
                                  scoring='neg_mean_absolute_error')

    return scores.mean()

def RF_params(preprocessor, X, y):

    #  Function to check the effect of different parameters at the model accuracy

    results= {}
    for i in range(200,410,100): #n_estimators
        for j in range(10,31,10): #learning_rate
            j=j/1000 # to convert to the correct learning rate value
            clf2 = model_and_clf(preprocessor, i, j)
            MAE_param = cross_validation(clf2, X, y)
            results["n_estimators" + str(i), "learning_rate="+ str(j)] = MAE_param

    results_sorted = {k: v for k, v in sorted(results.items(), key=lambda item: item[1])}

    return results_sorted

def create_X_test_pred(preprocessor, X_train, X_test, y_train, n_estimators, learning_rate):

    # Function to create the .csv document with predictions

    # Get predictions for X_test data
    preds = get_preds(preprocessor, X_train, X_test, y_train, n_estimators, learning_rate)

    # Create .csv document
    output = pd.DataFrame({'Id': X_test.index,
                           'SalePrice': preds})
    output.to_csv('Try3.csv', index=False)

    return

def Housing_Prices():
    # Give the filepaths for train and test documents
    X, y, X_test = data_load("Housing Prices/train.csv","Housing Prices/test.csv")

    # Understand the data - function plots each feature against the labels
    #plot = data_visualisation(X,y)

    # Remove features that only add noise-Year Sold & Month Sold: as we make a prediction about the SalePrice in general
    # and not about the SalePrice in a specific time in the future this adds only noise
    X.drop(["YrSold","MoSold"], axis=1, inplace=True)
    X_test.drop(["YrSold","MoSold"], axis=1, inplace=True)

    # Check features for missing values and make changes
    X, X_test = missing_values(X, X_test)

    # Select numeric and object columns
    numerical_cols = [cname for cname in X.columns if X[cname].dtype in ["int64", "float64"]]
    # if I wanted to get only low candinality cols for categorical_cols
    # categorical_cols = [cname for cname in X.columns if
    #                    X[cname].nunique() < 10 and
    #                    X[cname].dtype == "object"]
    categorical_cols = [cname for cname in X.columns if X[cname].dtype == "object"]

    # Break off validation set from training data
    X_train, X_valid, y_train, y_valid = train_test_split(X, y,
                                                          train_size=0.8, test_size=0.2,
                                                          random_state=0)

    # Define preprocessor
    preprocessor = preprocessing(numerical_cols,categorical_cols)

    # Define model and get classifier
    clf = model_and_clf(preprocessor)

    # Preprocessing of train & valid data, get predictions
    preds = get_preds(preprocessor,X_train, X_valid, y_train, 100, 0.1) # n_estimators, max_depth
    print("MAE_basic=", mean_absolute_error(y_valid, preds))

    # Use cross-validation to check consistency
    scores_cv = cross_validation(clf, X, y)
    print("Average MAE score=", scores_cv)

    # Find the best parameters for Random Forest Regressor
    MAE_diff_params = RF_params(preprocessor, X, y)
    print(MAE_diff_params)

    # Generate prediction file for submission
    pred_file = create_X_test_pred(preprocessor, X_train, X_test, y_train, 700, 0.02) # n_estimators, max_depth

    return pred_file

a=Housing_Prices()


