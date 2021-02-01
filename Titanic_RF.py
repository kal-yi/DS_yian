import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import (
    GridSearchCV,
    train_test_split,
)
from sklearn_pandas import DataFrameMapper
from sklearn.ensemble import RandomForestClassifier


def load_data(X_filepath, X_test_filepath):

    X_full = pd.read_csv(X_filepath, index_col="PassengerId")
    X_test = pd.read_csv(X_test_filepath, index_col="PassengerId")
    X_full.dropna(axis=0, subset=["Survived"], inplace=True)
    y = X_full.Survived
    X = X_full.drop(["Survived"], axis=1)

    return X, y, X_test, X_full


def get_fam_relationship(all_data2, fam_size, Age, Title, fam_number):

    if Age <= 17:
        return "child"

    elif fam_size == 1:
        return "alone"

    else:
        list_fam_Ages = []
        list_fam_Ages = all_data2[
            all_data2["family_numbered"] == fam_number
        ].Age.tolist()
        family_analysis = ["child" if x <= 17 else "adult" for x in list_fam_Ages]

        if (Title == "Mrs") or (Title == "Ms") & ("child" in family_analysis):
            return "mother"
        else:
            return "husband or sibling or father"


def data_preprocessing(X, X_test):

    # --------------------------------------------- Column "Age" ---------------------------------------------

    for dataframe in [X, X_test]:
        dataframe["Age"].fillna(85.0, inplace=True)

    # Replace approximate ages __.5 with 80
    for dataframe in [X, X_test]:
        dataframe.loc[(dataframe["Age"] % 1 == 0.5), ["Age"]] = 80

    # Treat exemptions
    X.loc[124, "Age"] = 32.5  # that person survived
    X.loc[335, "Age"] = 30  # she is a mother

    # --------------------------------------------- Column "Fare" ----------------------------------------------

    for dataframe in [X, X_test]:
        dataframe["Fare"].fillna(7.0, inplace=True)

    # --------------------------------------------- Column "Cabin" ---------------------------------------------

    for dataframe in [X, X_test]:
        dataframe["Cabin"].fillna(0, inplace=True)
        dataframe["Cabin"].replace(regex={"[a-zA-Z].*": "1"}, inplace=True)
        dataframe["Cabin"] = dataframe["Cabin"].astype("float64")

    # -------------------------------------------- Column "Embarked" --------------------------------------------

    for dataframe in [X, X_test]:
        dataframe["Embarked"].fillna("C", inplace=True)

    # --------------------------------------------- Column "Name" -----------------------------------------------

    frames = [X, X_test]
    all_data = pd.concat(frames)

    splitted = all_data["Name"].str.split(",")
    all_data["Surname"] = splitted.str[0]
    all_data["1st name"] = splitted.str[-1]
    splitted2 = all_data["1st name"].str.split(".")
    all_data["Title"] = splitted2.str[0].str.strip()

    all_data["Title"] = all_data["Title"].replace(
        [
            "Miss",
            "Dona",
            "Don",
            "Rev",
            "Mme",
            "Major",
            "Lady",
            "Sir",
            "Mlle",
            "Col",
            "Capt",
            "the Countess",
            "Jonkheer",
        ],
        [
            "Ms",
            "Ms",
            "Mr",
            "Mr",
            "Ms",
            "Mr",
            "Mrs",
            "Mr",
            "Ms",
            "Mr",
            "Mr",
            "Mrs",
            "Mr",
        ],
    )

    all_data["family_numbered"] = all_data.groupby(["Surname", "Ticket"]).ngroup()
    all_data["Family_size2"] = all_data["SibSp"] + all_data["Parch"] + 1

    # ----------------------------------------- Column "fam_relationship" -----------------------------------------
    # Column created to identify the children, mothers, etc.

    all_data2 = all_data.copy()
    all_data2["fam_relationship"] = "no_data"
    all_data2["fam_relationship"] = all_data2.apply(
        lambda column: get_fam_relationship(
            all_data2,
            column["Family_size2"],
            column["Age"],
            column["Title"],
            column["family_numbered"],
        ),
        axis=1,
    )

    # Finalize X and X_test
    all_data2 = all_data2.drop(
        [
            "Name",
            "1st name",
            "Surname",
            "Ticket",
            "family_numbered",
            "SibSp",
            "Parch",
        ],
        axis=1,
    )

    X = all_data2.iloc[:891, :]
    X_test = all_data2.iloc[891:, :]

    # ----------------------------------------- Scaling / Encoding -----------------------------------------

    column_tuples = [
        (["Pclass"], StandardScaler()),
        (["Sex"], OneHotEncoder()),
        (["Age"], StandardScaler()),
        (["Fare"], StandardScaler()),
        (["Cabin"], None),
        (["Embarked"], OneHotEncoder()),
        (["Title"], OneHotEncoder()),
        (["Family_size2"], None),
        (["fam_relationship"], OneHotEncoder()),
    ]

    mapper = DataFrameMapper(column_tuples, df_out=True)

    X_mapped = mapper.fit_transform(X)
    X_test_mapped = mapper.transform(X_test)

    X = X_mapped
    X_test = X_test_mapped

    return X, X_test


def apply_k_means(X, X_test):

    frames = [X, X_test]
    all_data = pd.concat(frames)

    kmeans = KMeans(n_clusters=30, init="k-means++", random_state=42)
    y_kmeans = kmeans.fit_predict(all_data)

    all_data["kmeans"] = y_kmeans

    X = all_data.iloc[:891, :]
    X_test = all_data.iloc[891:, :]

    return X, X_test


def hyperparams_selection(X, y):

    grid_params = {
        "n_estimators": range(305, 310, 5),
        "criterion": ["gini"],  # ,"entropy"]
        "max_depth": range(10, 11, 1),
        "max_features": range(14, 15, 1),
        "max_leaf_nodes": range(11, 12, 1),
    }

    model_grid = GridSearchCV(
        estimator=RandomForestClassifier(),
        param_grid=grid_params,
        scoring="accuracy",
        cv=5,
        n_jobs=-1,
        verbose=True,
    )

    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.2, random_state=4
    )

    model_grid.fit(X_train, y_train)
    Accuracy_best_score_train = model_grid.best_score_
    best_parameters = model_grid.best_params_

    return best_parameters


def model(X, y, best_parameters):

    rf_model = RandomForestClassifier(**best_parameters)

    return rf_model.fit(X, y)


def save_Y_test_pred(model, X_test):

    preds = model.predict(X_test)

    output = pd.DataFrame({"PassengerId": X_test.index, "Survived": preds})
    output.to_csv("Titanic/Titanic_Pycharm_RF.csv", index=False)


def Titanic_RF():

    X, y, X_test, X_full = load_data("Titanic/train.csv", "Titanic/test.csv")

    X, X_test = data_preprocessing(X, X_test)

    X, X_test = apply_k_means(X, X_test)

    rf_hyperparams = hyperparams_selection(X, y)

    rf_model = model(X, y, rf_hyperparams)

    save_Y_test_pred(rf_model, X_test)


if __name__ == "__main__":
    Titanic_RF()
