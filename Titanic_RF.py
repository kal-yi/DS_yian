import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
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


def fam_relationship(all_data2, fam_size, SibSp, Parch, fam_number, index_num):

    # Avoid duplicating work - relationship has been defined when checking the family
    if all_data2["fam_relationship"].loc[index_num] != "no_data":
        return all_data2["fam_relationship"].loc[index_num]

    elif all_data2["Age"].loc[index_num] <= 15:
        all_data2["fam_relationship"].loc[index_num] = "child"
        return all_data2["fam_relationship"].loc[index_num]

    # 1 person
    elif fam_size == 1:
        if (SibSp == 0) & (Parch == 0):
            all_data2["fam_relationship"].loc[index_num] = "alone"
            return all_data2["fam_relationship"].loc[index_num]

        elif (SibSp == 0) & (Parch == 1):
            if (all_data2["Title"].loc[index_num] == "Mrs") & (
                all_data2["Age"].loc[index_num] <= 50
            ):
                all_data2["fam_relationship"].loc[index_num] = "mother"
                return all_data2["fam_relationship"].loc[index_num]
            elif all_data2["Age"].loc[index_num] <= 17:
                all_data2["fam_relationship"].loc[index_num] = "child"
                return all_data2["fam_relationship"].loc[index_num]
            else:
                all_data2["fam_relationship"].loc[index_num] = "husband or sibling"
                return all_data2["fam_relationship"].loc[index_num]
        else:
            all_data2["fam_relationship"].loc[index_num] = "husband or sibling"
            return all_data2["fam_relationship"].loc[index_num]

    # 2 people
    elif fam_size == 2:
        list_fam_2 = []
        list_fam_2 = all_data2[
            all_data2["family_numbered"] == fam_number
        ].index.tolist()  # get the indexes of all family members

        if (all_data2["SibSp"].loc[list_fam_2[0]] == 1) & (
            all_data2["SibSp"].loc[list_fam_2[1]] == 1
        ):
            all_data2["fam_relationship"].loc[list_fam_2[0]] = "husband or sibling"
            all_data2["fam_relationship"].loc[list_fam_2[1]] = "husband or sibling"
            return all_data2["fam_relationship"].loc[index_num]

        elif (all_data2["Parch"].loc[list_fam_2[0]] == 1) & (
            all_data2["Parch"].loc[list_fam_2[1]] == 1
        ):
            if (
                all_data2["Age"].loc[list_fam_2[0]]
                > all_data2["Age"].loc[list_fam_2[1]]
            ):
                if all_data2["Title"].loc[list_fam_2[0]] == "Mrs":
                    all_data2["fam_relationship"].loc[list_fam_2[0]] = "mother"
                    all_data2["fam_relationship"].loc[list_fam_2[1]] = "child"
                    return all_data2["fam_relationship"].loc[
                        index_num
                    ]  # mother or child
                else:
                    all_data2["fam_relationship"].loc[list_fam_2[0]] = "father"
                    all_data2["fam_relationship"].loc[list_fam_2[1]] = "child"
                    return all_data2["fam_relationship"].loc[
                        index_num
                    ]  # father or child
            else:
                if all_data2["Title"].loc[list_fam_2[1]] == "Mrs":
                    all_data2["fam_relationship"].loc[list_fam_2[1]] = "mother"
                    all_data2["fam_relationship"].loc[list_fam_2[0]] = "child"
                    return all_data2["fam_relationship"].loc[
                        index_num
                    ]  # mother or child
                else:
                    all_data2["fam_relationship"].loc[list_fam_2[1]] = "father"
                    all_data2["fam_relationship"].loc[list_fam_2[0]] = "child"
                    return all_data2["fam_relationship"].loc[
                        index_num
                    ]  # father or child

        elif (SibSp == 0) & (Parch == 0):
            all_data2["fam_relationship"].loc[index_num] = "alone"
            return all_data2["fam_relationship"].loc[index_num]

        else:
            all_data2["fam_relationship"].loc[index_num] = "husband or sibling"
            return all_data2["fam_relationship"].loc[index_num]

    # 3 people
    elif fam_size == 3:
        list_fam_3 = []
        list_fam_3 = all_data2[
            all_data2["family_numbered"] == fam_number
        ].index.tolist()  # get the indexes of all family members

        if (all_data2["SibSp"].loc[index_num] == 0) & (
            all_data2["Parch"].loc[index_num] == 2
        ):
            list_fam_3.remove(index_num)
            if (
                (all_data2["Age"].loc[index_num] + float(12))
                <= all_data2["Age"].loc[list_fam_3[0]]
            ) & (
                (all_data2["Age"].loc[index_num] + float(12))
                <= all_data2["Age"].loc[list_fam_3[1]]
            ):
                all_data2["fam_relationship"].loc[index_num] = "child"
                if all_data2["Title"].loc[list_fam_3[0]] == "Mrs":
                    all_data2["fam_relationship"].loc[list_fam_3[0]] = "mother"
                    all_data2["fam_relationship"].loc[list_fam_3[1]] = "father"
                    return all_data2["fam_relationship"].loc[index_num]  # child
                else:
                    all_data2["fam_relationship"].loc[list_fam_3[0]] = "father"
                    all_data2["fam_relationship"].loc[list_fam_3[1]] = "mother"
                    return all_data2["fam_relationship"].loc[index_num]  # child

            elif all_data2["Age"].loc[index_num] <= 17:
                all_data2["fam_relationship"].loc[index_num] = "child"
                return all_data2["fam_relationship"].loc[
                    index_num
                ]  # not 100% accurate (it could be spend more time on it)
            else:
                if all_data2["Title"].loc[index_num] == "Mrs":
                    all_data2["fam_relationship"].loc[index_num] = "mother"
                    return all_data2["fam_relationship"].loc[index_num]
                else:
                    all_data2["fam_relationship"].loc[index_num] = "father"
                    return all_data2["fam_relationship"].loc[index_num]

        elif (all_data2["SibSp"].loc[index_num] == 1) & (
            all_data2["Parch"].loc[index_num] == 1
        ):
            if all_data2["Age"].loc[index_num] <= 17:
                all_data2["fam_relationship"].loc[index_num] = "child"
                return all_data2["fam_relationship"].loc[
                    index_num
                ]  # not 100% accurate (it could be spend more time on it)

            elif all_data2["Title"].loc[index_num] == "Mrs":
                all_data2["fam_relationship"].loc[index_num] = "mother"
                return all_data2["fam_relationship"].loc[
                    index_num
                ]  # not 100% accurate (it could be spend more time on it)

            else:
                all_data2["fam_relationship"].loc[index_num] = "husband or sibling"
                return all_data2["fam_relationship"].loc[index_num]  # sibling

        else:
            all_data2["fam_relationship"].loc[index_num] = "husband or sibling"
            return all_data2["fam_relationship"].loc[index_num]

    # 4 people
    elif fam_size == 4:
        list_fam_4 = []
        list_fam_4 = all_data2[
            all_data2["family_numbered"] == fam_number
        ].index.tolist()  # get the indexes of all family members

        if (all_data2["SibSp"].loc[index_num] == 3) & (
            all_data2["Parch"].loc[index_num] == 0
        ):
            all_data2["fam_relationship"].loc[index_num] = "husband or sibling"
            return all_data2["fam_relationship"].loc[index_num]  # sibling

        elif (all_data2["SibSp"].loc[index_num] == 2) & (
            all_data2["Parch"].loc[index_num] == 1
        ):
            all_data2["fam_relationship"].loc[index_num] = "child"
            return all_data2["fam_relationship"].loc[index_num]

        elif (all_data2["SibSp"].loc[index_num] == 0) & (
            all_data2["Parch"].loc[index_num] == 3
        ):
            if (all_data2["Title"].loc[index_num] == "Mrs") or (
                all_data2["Title"].loc[index_num] == "Ms"
            ):
                all_data2["fam_relationship"].loc[index_num] = "mother"
                return all_data2["fam_relationship"].loc[index_num]
            else:
                all_data2["fam_relationship"].loc[index_num] = "father"
                return all_data2["fam_relationship"].loc[index_num]

        elif (all_data2["Title"].loc[index_num] == "Mrs") & (
            all_data2["Age"].loc[index_num] <= 50
        ):
            all_data2["fam_relationship"].loc[index_num] = "mother"
            return all_data2["fam_relationship"].loc[index_num]

        else:
            all_data2["fam_relationship"].loc[index_num] = "husband or sibling"
            return all_data2["fam_relationship"].loc[index_num]  # sibling

    # 5 people
    elif fam_size == 5:

        if (all_data2["SibSp"].loc[index_num] == 3) & (
            all_data2["Parch"].loc[index_num] == 1
        ):
            all_data2["fam_relationship"].loc[index_num] = "child"
            return all_data2["fam_relationship"].loc[index_num]

        elif (all_data2["SibSp"].loc[index_num] == 2) & (
            all_data2["Parch"].loc[index_num] == 2
        ):
            all_data2["fam_relationship"].loc[index_num] = "child"
            return all_data2["fam_relationship"].loc[index_num]

        elif (
            (all_data2["SibSp"].loc[index_num] == 1)
            & (all_data2["Parch"].loc[index_num] == 3)
        ) or (
            (all_data2["SibSp"].loc[index_num] == 0)
            & (all_data2["Parch"].loc[index_num] == 4)
        ):
            if (
                (all_data2["Title"].loc[index_num] == "Mrs")
                or (all_data2["Title"].loc[index_num] == "Ms")
            ) & (all_data2["Age"].loc[index_num] <= 50):
                all_data2["fam_relationship"].loc[index_num] = "mother"
                return all_data2["fam_relationship"].loc[index_num]
            else:
                all_data2["fam_relationship"].loc[index_num] = "husband or sibling"
                return all_data2["fam_relationship"].loc[index_num]  # sibling

        else:
            all_data2["fam_relationship"].loc[index_num] = "husband or sibling"
            return all_data2["fam_relationship"].loc[index_num]  # sibling

    # 6 people
    elif fam_size == 6:

        if (
            (all_data2["SibSp"].loc[index_num] == 1)
            & (all_data2["Parch"].loc[index_num] == 4)
        ) or (
            (all_data2["SibSp"].loc[index_num] == 0)
            & (all_data2["Parch"].loc[index_num] == 5)
        ):
            if (
                (all_data2["Title"].loc[index_num] == "Mrs")
                or (all_data2["Title"].loc[index_num] == "Ms")
            ) & (all_data2["Age"].loc[index_num] <= 50):
                all_data2["fam_relationship"].loc[index_num] = "mother"
                return all_data2["fam_relationship"].loc[index_num]
            else:
                all_data2["fam_relationship"].loc[index_num] = "father"
                return all_data2["fam_relationship"].loc[index_num]

        else:
            all_data2["fam_relationship"].loc[index_num] = "husband or sibling"
            return all_data2["fam_relationship"].loc[index_num]  # sibling

    # 7 people
    elif fam_size == 7:

        if (
            (all_data2["SibSp"].loc[index_num] == 1)
            & (all_data2["Parch"].loc[index_num] == 5)
        ) or (
            (all_data2["SibSp"].loc[index_num] == 0)
            & (all_data2["Parch"].loc[index_num] == 6)
        ):
            if (
                (all_data2["Title"].loc[index_num] == "Mrs")
                or (all_data2["Title"].loc[index_num] == "Ms")
            ) & (all_data2["Age"].loc[index_num] <= 50):
                all_data2["fam_relationship"].loc[index_num] = "mother"
                return all_data2["fam_relationship"].loc[index_num]
            else:
                all_data2["fam_relationship"].loc[index_num] = "father"
                return all_data2["fam_relationship"].loc[index_num]

        else:
            all_data2["fam_relationship"].loc[index_num] = "husband or sibling"
            return all_data2["fam_relationship"].loc[index_num]

    # 8 people
    elif fam_size == 8:

        if (
            (all_data2["SibSp"].loc[index_num] == 1)
            & (all_data2["Parch"].loc[index_num] == 6)
        ) or (
            (all_data2["SibSp"].loc[index_num] == 0)
            & (all_data2["Parch"].loc[index_num] == 7)
        ):
            if (
                (all_data2["Title"].loc[index_num] == "Mrs")
                or (all_data2["Title"].loc[index_num] == "Ms")
            ) & (all_data2["Age"].loc[index_num] <= 50):
                all_data2["fam_relationship"].loc[index_num] = "mother"
                return all_data2["fam_relationship"].loc[index_num]
            else:
                all_data2["fam_relationship"].loc[index_num] = "father"
                return all_data2["fam_relationship"].loc[index_num]

        else:
            all_data2["fam_relationship"].loc[index_num] = "husband or sibling"
            return all_data2["fam_relationship"].loc[index_num]  # sibling

    # >=9 people
    elif fam_size >= 9:
        all_data2["fam_relationship"].loc[index_num] = "husband or sibling"
        return all_data2["fam_relationship"].loc[index_num]

    elif (SibSp == 0) & (Parch == 0):
        all_data2["fam_relationship"].loc[index_num] = "alone"
        return 0, "alone"

    else:
        all_data2["fam_relationship"].loc[index_num] = "I dont know"
        return all_data2["fam_relationship"].loc[index_num]


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
    all_data["Family_size"] = all_data.family_numbered.map(
        all_data.groupby("family_numbered").size()
    )  # family size calculated
    all_data["Family_size2"] = (
        all_data["SibSp"] + all_data["Parch"] + 1
    )  # true family size

    # ----------------------------------------- Column "fam_relationship" -----------------------------------------
    # Column created to identify the children, mothers, fathers, etc.

    all_data2 = all_data.copy()
    all_data2["fam_relationship"] = "no_data"
    all_data2["fam_relationship"] = all_data2.apply(
        lambda column: fam_relationship(
            all_data2,
            column["Family_size"],
            column["SibSp"],
            column["Parch"],
            column["family_numbered"],
            column.name,
        ),
        axis=1,
    )

    # Treat exemptions correctly
    mother_list = [86, 137, 248, 335, 541, 581, 601, 775, 996, 1045, 1133, 1222]
    for mother in mother_list:
        all_data2["fam_relationship"].loc[mother] = "mother"

    child_list = [418, 944]
    for child in child_list:
        all_data2["fam_relationship"].loc[child] = "child"

    all_data2["fam_relationship"].loc[146] = "husband or sibling"

    # Finalize X and X_test
    all_data2 = all_data2.drop(
        [
            "Name",
            "1st name",
            "Surname",
            "Ticket",
            "family_numbered",
            "Family_size2",
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
        (["Family_size"], None),
        (["fam_relationship"], OneHotEncoder()),
    ]

    mapper = DataFrameMapper(column_tuples, df_out=True)

    X_mapped = mapper.fit_transform(X)
    X_test_mapped = mapper.transform(X_test)

    X = X_mapped
    X_test = X_test_mapped

    return X, X_test


def hyperparams_selection(X, y):

    grid_params = {
        "n_estimators": range(340, 341, 5),
        "criterion": ["gini"],  # ,"entropy"]
        "max_depth": range(10, 13, 1),
        "max_features": range(10, 13, 1),
        "max_leaf_nodes": range(18, 20, 1),
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

    RF_model = RandomForestClassifier(**best_parameters)

    return RF_model.fit(X, y)


def save_Y_test_pred(model, X_test):

    preds = model.predict(X_test)

    output = pd.DataFrame({"PassengerId": X_test.index, "Survived": preds})
    output.to_csv("Titanic/Titanic_Pycharm_RF.csv", index=False)


def Titanic_RF():

    X, y, X_test, X_full = load_data("Titanic/train.csv", "Titanic/test.csv")

    X, X_test = data_preprocessing(X, X_test)

    RF_hyperparams = hyperparams_selection(X, y)

    RF_model = model(X, y, RF_hyperparams)

    pred_file = save_Y_test_pred(RF_model, X_test)

    return pred_file


if __name__ == "__main__":
    Titanic_RF()
