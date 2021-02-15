import numpy as np
import pandas as pd
from sklearn_pandas import DataFrameMapper
from datetime import datetime, date
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import (
    TimeSeriesSplit,
    GridSearchCV,
)
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import random


def load_data(x_filepath):

    x_full = pd.read_csv(x_filepath)
    x_full.sort_values(["search_date", "search_id"], inplace=True)
    x_full.drop(["Unnamed: 15", "Unnamed: 16", "Unnamed: 17"], axis=1, inplace=True)

    return x_full


def data_preprocessing(x1):

    # ---------------------------------------------- fillna -------------------------------------------------

    for column in ["hotel_feature_1", "hotel_feature_2"]:
        x1[column].fillna(0, inplace=True)

    # -------------------------------------------- Date format ----------------------------------------------

    for column in ["search_date", "arrival", "departure"]:
        x1[column] = pd.to_datetime(x1[column], format="%Y%m%d")

    # ----------------------------------------- Create new features -----------------------------------------

    # Identify the day of the month the search was carried -- all searches on January 2015
    x1["search_dayofmonth"] = x1["search_date"].dt.day

    # Identify the day of the week that the search was carried
    x1["search_dayofweek"] = x1["search_date"].dt.dayofweek

    # Number of days from "search_date" till start of trip ("arrival")
    x1["search_trip_period"] = (x1["arrival"] - x1["search_date"]).dt.days

    # Trip duration
    x1["trip_duration"] = (x1["departure"] - x1["arrival"]).dt.days

    # Number of times that each instance (row) searched per day
    x1["searches_num"] = x1.groupby(["search_id", "search_dayofmonth"])[
        "search_dayofmonth"
    ].transform("size")

    # Count the identical searches -- Identical rows numbered
    x1["identical_searches"] = x1.groupby(x1.columns.tolist())["search_date"].transform(
        "size"
    )

    # Drop duplicate rows
    x1 = x1.drop_duplicates(keep="last")
    x1 = x1.reset_index(drop=True)

    # Total number of people travelling
    x1["tot_people_travelling"] = x1["num_adults"] + x1["num_children"]

    # Hotel price per person
    x1["hotel_price_per_person"] = x1["hotel_price"] / x1["tot_people_travelling"]

    # Travel starting day (Mon _ Tues etc.)
    x1["arrival_dayofweek"] = x1["arrival"].dt.dayofweek

    # Start Travel week   # Add 52 weeks to people travelling in 2016
    x1["arrival_weekofyear"] = x1["arrival"].dt.isocalendar().week
    x1["arrival_year"] = x1["arrival"].dt.year
    x1["arrival_weekofyear"] = x1["arrival_weekofyear"].mask(
        x1["arrival_year"] == 2016, x1["arrival_weekofyear"] + 52
    )

    # Number of people looked to visit each hotel every week of the year
    x1["hotel_arrival_week_total"] = x1.groupby(
        ["arrival_weekofyear", "search_dayofmonth", "hotel_id"]
    )["tot_people_travelling"].transform("sum")

    # Number of times each hotel selected for every week of the year each day
    x1["hotel_search_arrivals"] = x1.groupby(
        ["hotel_id", "search_dayofmonth", "arrival_weekofyear"]
    )["search_dayofmonth"].transform("size")

    # Cumulative number of people want to visit all hotels every week of the year
    x1["cum_sum_week_0"] = x1.groupby(["search_dayofmonth", "arrival_weekofyear"])[
        "tot_people_travelling"
    ].transform("cumsum")
    x1["cum_sum_week"] = x1.groupby(["search_dayofmonth", "arrival_weekofyear"])[
        "cum_sum_week_0"
    ].transform(
        "max"
    )  # keep the max value of each final group

    # -------------------------------------------------------------------------------------------
    # The following A, B, C were not used in the final model as they were not improving the final result
    # I think that the following features would have worked if the data had included more bookings or longer period of data acquisition

    # ---------------------------------------  A  ----------------------------------------------
    # Number of people looked to travel each day (search day)
    x1["tot_people"] = x1.groupby(["search_dayofmonth"])[
        "tot_people_travelling"
    ].transform("sum")

    # Number of people looked to travel to each hotel each day (search day)
    x1["tot_people_hotel"] = x1.groupby(["hotel_id", "search_dayofmonth"])[
        "tot_people_travelling"
    ].transform("sum")

    # Ratio: Number of people looked to travel to each hotel / Total number of people looked to travel --- EACH DAY
    x1["people_ratio_hotel"] = x1["tot_people_hotel"] / x1["tot_people"]
    # ------------------------------------------------------------------------------------------

    # ---------------------------------------  B  ----------------------------------------------
    # Number of times each hotel selected per day
    x1["hotel_searches_per_day"] = x1.groupby(["hotel_id", "search_dayofmonth"])[
        "search_dayofmonth"
    ].transform("size")

    # Total searches per day
    x1["total_searches_per_day"] = x1.groupby(["search_dayofmonth"])[
        "search_dayofmonth"
    ].transform("size")

    # Ratio: total hotel searches / total searches --- EACH DAY
    x1["searches_ratio_hotel"] = (
        x1["hotel_searches_per_day"] / x1["total_searches_per_day"]
    )
    # ------------------------------------------------------------------------------------------

    # ---------------------------------------  C  ----------------------------------------------
    # Number of times each hotel selected for every week of the year each day
    x1["hotel_search_arrivals"] = x1.groupby(
        ["hotel_id", "search_dayofmonth", "arrival_weekofyear"]
    )["search_dayofmonth"].transform("size")

    # Cumulative number of people want to visit all hotels every week of the year
    x1["cum_sum_week_0"] = x1.groupby(["search_dayofmonth", "arrival_weekofyear"])[
        "tot_people_travelling"
    ].transform("cumsum")
    x1["cum_sum_week"] = x1.groupby(["search_dayofmonth", "arrival_weekofyear"])[
        "cum_sum_week_0"
    ].transform(
        "max"
    )  # keep the max value of each final group

    # Cumulative number of people want to visit each hotel every week of the year
    x1["cum_sum_hotel_week_0"] = x1.groupby(["hotel_id", "arrival_weekofyear"])[
        "tot_people_travelling"
    ].transform("cumsum")
    x1["cum_sum_hotel_week"] = x1.groupby(
        ["search_dayofmonth", "hotel_id", "arrival_weekofyear"]
    )["cum_sum_hotel_week_0"].transform(
        "max"
    )  # keep the max value of each final group

    # Calculate the ratio: people looked to visit the specific week / total people want to visit THIS hotel
    x1["people_ratio_hotel_week"] = x1["cum_sum_hotel_week"] / x1["cum_sum_week"]

    # Calculate the ratio: people looked to visit the specific week / total people want to visit all time
    x1["people_ratio_week"] = x1["cum_sum_week"] / x1["tot_people"]
    # ------------------------------------------------------------------------------------------

    def get_trip_purpose(adults_num, children_num):

        if (adults_num == 1) & (children_num == 0):
            return 0  # travelling alone - probably for work

        elif (adults_num >= 2) & (children_num == 0):
            return 1  # group travelling for business purposes or couple for holidays

        else:
            return 2  # family/ies travelling for holidays

    # Identify the purpose of the trip
    x01 = x1.copy()
    x1["trip_purpose"] = x01.apply(
        lambda column: get_trip_purpose(column["num_adults"], column["num_children"]),
        axis=1,
    )

    return x1


def downsampling(x):

    xA = x.copy()

    # Total number of hotels: 355
    # Number of hotels that did NOT get a booking: 252

    # Create a list with the hotels that did not get a booking
    xA1 = x.copy()
    xA1["hotel_booked"] = xA.groupby(["hotel_id"])["booked"].transform("sum")
    xA1 = xA1[xA1["hotel_booked"] == 0]
    hotel_list = xA1.hotel_id.unique()

    days_list = xA.search_dayofmonth.unique()

    # Randomly remove rows of hotel searches that received no bookings
    for hotel in random.choices(hotel_list, k=100):
        random_days_list = random.choices(days_list, k=8)
        indexNums = xA[
            (
                (xA["search_dayofmonth"] == random_days_list[0])
                | (xA["search_dayofmonth"] == random_days_list[1])
                | (xA["search_dayofmonth"] == random_days_list[2])
                | (xA["search_dayofmonth"] == random_days_list[3])
                | (xA["search_dayofmonth"] == random_days_list[4])
                | (xA["search_dayofmonth"] == random_days_list[5])
                | (xA["search_dayofmonth"] == random_days_list[6])
                | (xA["search_dayofmonth"] == random_days_list[7])
            )
            & (xA["hotel_id"] == hotel)
        ].index
        xA.drop(indexNums, inplace=True)

    # 3% reduction on the number of rows
    # Generally data removal reduces the roc_auc_score

    xA = xA.reset_index(drop=True)

    return xA


def train_test_split(x1):

    # Train
    x = x1[x1["search_dayofmonth"] <= 23]
    x = downsampling(x)
    y = x.pop("booked")

    # Test
    x_test = x1[x1["search_dayofmonth"] >= 24]  # 10% of the dataset
    y_test = x_test.pop("booked")

    print(x.shape)

    return x, y, x_test, y_test


def get_timeseries_indices(x, split_num, test_size_num):

    # Function gives the indices for the timeseries according to the search date

    date_series = x["search_dayofmonth"]

    min_date = date_series.min()
    max_date = date_series.max()

    # no data given for dates: 6, 19 and 23 January 2015
    dates = list(range(min_date, max_date + 2))

    splitter = TimeSeriesSplit(n_splits=split_num, test_size=test_size_num)
    date_splits = list(splitter.split(dates))

    # Get the split_indices
    split_indices = list()
    for train_dates, valid_dates in date_splits:
        train_indices = [i for i in date_series[date_series.isin(train_dates)].index]
        valid_indices = [i for i in date_series[date_series.isin(valid_dates)].index]
        split_indices.append((train_indices, valid_indices))

    return split_indices


def dataframe_preparation(x):

    column_tuples = [
        (["is_promo"], None),
        (["hotel_feature_1"], StandardScaler()),
        (["hotel_feature_2"], StandardScaler()),
        (["hotel_feature_3"], StandardScaler()),
        (["hotel_feature_4"], StandardScaler()),
        (["hotel_feature_5"], StandardScaler()),
        (["search_dayofweek"], StandardScaler()),
        (["search_trip_period"], StandardScaler()),
        (["trip_duration"], StandardScaler()),
        (["searches_num"], StandardScaler()),
        (["identical_searches"], StandardScaler()),
        (["trip_purpose"], OneHotEncoder()),
        (["tot_people_travelling"], StandardScaler()),
        (["hotel_price_per_person"], StandardScaler()),
        (["arrival_dayofweek"], StandardScaler()),
        (["hotel_arrival_week_total"], StandardScaler()),
    ]

    mapper = DataFrameMapper(column_tuples, df_out=True)

    x_mapped = mapper.fit_transform(x)

    return x_mapped, mapper


def hyperparams_selection(x, y, split_indices):

    grid_params = {
        "learning_rate": np.arange(0.07, 0.08, 0.01),
        "n_estimators": range(85, 90, 5)
        # ,"criterion": ["friedman_mse"]#,"mse","mae"]
        # ,"min_samples_split": range(1, 3, 1)
        # ,"max_depth": range(10, 11, 2)
        # ,"max_leaf_nodes": range(20, 21, 2)
        ,
        "random_state": [4],
    }

    model_grid = GridSearchCV(
        estimator=GradientBoostingClassifier(),
        param_grid=grid_params,
        scoring="roc_auc",
        cv=split_indices,
        n_jobs=-1,
        verbose=True,
    )

    model_grid.fit(x, y)

    roc_auc_best_score_train = model_grid.best_score_
    best_parameters = model_grid.best_params_

    print("Best Estimator: \n{}\n".format(model_grid.best_estimator_))
    print("Best Parameters: \n{}\n".format(model_grid.best_params_))
    print("Best Test Score: \n{}\n".format(model_grid.best_score_))

    return best_parameters


def model(x, y, best_parameters):

    gbc_model = GradientBoostingClassifier(**best_parameters)

    return gbc_model.fit(x, y)


def roc_auc_x_test(gbc_model, x_test, y_test, mapper):

    x_test_mapped = mapper.transform(x_test)

    y_pred = gbc_model.predict_proba(x_test_mapped)[:, 1]

    roc_auc_test = roc_auc_score(y_test, y_pred)

    print("Roc_auc_test_with_best_params: \n{}\n".format(roc_auc_test))
    # Result: 0.7452726522056722


def gbc_feature_importances(gbc_model, x):

    # Function to plot the feature importances

    features = x.columns
    importances = gbc_model.feature_importances_
    indices = np.argsort(importances)

    plt.title("Feature Importances")
    plt.barh(range(len(indices)), importances[indices], color="b", align="center")
    plt.yticks(range(len(indices)), [features[i] for i in indices])
    plt.xlabel("Relative Importance")
    plt.show()


def expedia():

    x_full = load_data("Expedia\case_study_data.csv")

    x_full = data_preprocessing(x_full)

    x, y, x_test, y_test = train_test_split(x_full)

    split_indices = get_timeseries_indices(x, split_num=3, test_size_num=5)

    x, mapper = dataframe_preparation(x)

    gbc_hyperparams = hyperparams_selection(x, y, split_indices)

    gbc_model = model(x, y, gbc_hyperparams)

    roc_auc_x_test(gbc_model, x_test, y_test, mapper)

    gbc_feature_importances(gbc_model, x)


if __name__ == "__main__":
    expedia()
