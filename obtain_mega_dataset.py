import datasets
import pandas as pd


def convert_dataframe_quail(df: pd.DataFrame, type=None):
    df_list = []
    for index, row in df.iterrows():
        for idx, option in enumerate(row["answers"]):
            data = {
                "question": row["question"],
                "answer": option,
                "context": row["context"],
            }

            if idx == row["correct_answer_id"]:
                data["correct"] = 1
            else:
                data["correct"] = 0

            df_list.append(data)

    return pd.DataFrame(df_list)


def convert_dataframe_race(df: pd.DataFrame, type=None):
    df_list = []
    label_map = {"A": 0, "B": 1, "C": 2, "D": 3}

    for index, row in df.iterrows():
        for idx, option in enumerate(row["options"]):
            data = {
                "question": row["question"],
                "answer": option,
                "context": row["article"],
            }

            if idx == label_map.get(row["answer"], -1):
                data["correct"] = 1
            else:
                data["correct"] = 0

            df_list.append(data)

    return pd.DataFrame(df_list)


def convert_dataframe_translated_quail(df: pd.DataFrame, type=None):
    df_list = []
    for index, row in df.iterrows():
        for idx, option in enumerate(row["translated_answers"]):
            data = {
                "question": row["translated_question"],
                "answer": option,
                "context": row["translated_context"],
            }

            if idx == row["correct_answer_id"]:
                data["correct"] = 1
            else:
                data["correct"] = 0

            df_list.append(data)

    return pd.DataFrame(df_list)


def convert_dataframe_translated_race(df: pd.DataFrame, type=None):
    df_list = []
    label_map = {"A": 0, "B": 1, "C": 2, "D": 3}

    for index, row in df.iterrows():
        for idx, option in enumerate(row["translated_answers"]):
            data = {
                "question": row["translated_question"],
                "answer": option,
                "context": row["translated_context"],
            }

            if idx == label_map.get(row["answer"], -1):
                data["correct"] = 1
            else:
                data["correct"] = 0

            df_list.append(data)

    return pd.DataFrame(df_list)


if __name__ == "__main__":

    mode = "MEGA-SPANISH-SINGLE"
    # obtaining RACE + QUAIL + RECORES datasets and concatenating
    # spanish datasets
    if mode == "MEGA-SPANISH-SINGLE":
        df_t_quail_train = pd.read_csv("data/translated/translated_quail_train.csv")
        df_t_quail_train["translated_answers"] = df_t_quail_train[
            "translated_answers"
        ].str.split("|")
        df_t_quail_val = pd.read_csv("data/translated/translated_quail_val.csv")
        df_t_quail_val["translated_answers"] = df_t_quail_val[
            "translated_answers"
        ].str.split("|")
        df_t_quail_test = pd.read_csv("data/translated/translated_quail_test.csv")
        df_t_quail_test["translated_answers"] = df_t_quail_test[
            "translated_answers"
        ].str.split("|")

        df_t_race_train = pd.read_csv("data/translated/final_translated_race_train.csv")
        df_t_race_train["translated_answers"] = df_t_race_train[
            "translated_answers"
        ].str.split(",")
        df_t_race_val = pd.read_csv("data/translated/final_translated_race_val.csv")
        df_t_race_val["translated_answers"] = df_t_race_val[
            "translated_answers"
        ].str.split(",")
        df_t_race_test = pd.read_csv("data/translated/final_translated_race_test.csv")
        df_t_race_test["translated_answers"] = df_t_race_test[
            "translated_answers"
        ].str.split(",")

        df_recores_train = pd.read_csv("data/original/recores_train.csv")
        df_recores_train = df_recores_train.drop(columns=['text', 'reason'])
        df_recores_val = df_recores_val = pd.read_csv("data/original/recores_val.csv")
        df_recores_val.drop(columns=['text', 'reason'])
        df_recores_test = df_recores_test = pd.read_csv("data/original/recores_test.csv")
        df_recores_test.drop(columns=['text', 'reason'])

        # binarirze
        df_t_quail_train = convert_dataframe_translated_quail(df_t_quail_train)
        df_t_quail_val = convert_dataframe_translated_quail(df_t_quail_val)
        df_t_quail_test = convert_dataframe_translated_quail(df_t_quail_test)

        df_t_race_train = convert_dataframe_translated_race(df_t_race_train)
        df_t_race_val = convert_dataframe_translated_race(df_t_race_val)
        df_t_race_test = convert_dataframe_translated_race(df_t_race_test)

        # concatenate and save
        frames = [df_t_quail_train, df_t_race_train, df_recores_train]
        result_train = pd.concat(frames)

        frames = [df_t_quail_val, df_t_race_val, df_recores_val]
        result_val = pd.concat(frames)

        frames = [df_t_quail_test, df_t_race_test, df_recores_test]
        result_test = pd.concat(frames)

        result_train.to_csv("./data/translated/mega_spanish_train.csv", index=False)
        result_val.to_csv("./data/translated/mega_spanish_val.csv", index=False)
        result_test.to_csv("./data/translated/mega_spanish_test.csv", index=False)

    if mode == "RACE-SINGLE-SPANISH":
        df_t_race_train = pd.read_csv("data/translated/final_translated_race_train.csv")
        df_t_race_train["translated_answers"] = df_t_race_train[
            "translated_answers"
        ].str.split(",")
        df_t_race_val = pd.read_csv("data/translated/final_translated_race_val.csv")
        df_t_race_val["translated_answers"] = df_t_race_val[
            "translated_answers"
        ].str.split(",")
        df_t_race_test = pd.read_csv("data/translated/final_translated_race_test.csv")
        df_t_race_test["translated_answers"] = df_t_race_test[
            "translated_answers"
        ].str.split(",")

        df_t_race_train = convert_dataframe_translated_race(df_t_race_train)
        df_t_race_val = convert_dataframe_translated_race(df_t_race_val)
        df_t_race_test = convert_dataframe_translated_race(df_t_race_test)

        df_t_race_train.to_csv("./data/translated/race_single_spanish_train.csv", index=False)
        df_t_race_val.to_csv("./data/translated/race_single_spanish_val.csv", index=False)
        df_t_race_test.to_csv("./data/translated/race_single_spanish_test.csv", index=False)


    if mode == "RACE-SINGLE":
        race = datasets.load_dataset("race", "all")
        df_race_train = race["train"].to_pandas()
        df_race_val = race["validation"].to_pandas()
        df_race_test = race["test"].to_pandas()

        df_race_train = convert_dataframe_race(df_race_train)
        df_race_val = convert_dataframe_race(df_race_val)
        df_race_test = convert_dataframe_race(df_race_test)

        df_race_train.to_csv("./data/original/race_single_train.csv", index=False)
        df_race_val.to_csv("./data/original/race_single_val.csv", index=False)
        df_race_test.to_csv("./data/original/race_single_test.csv", index=False)

    if mode == "MEGA-ENGLISH-SINGLE":
        race = datasets.load_dataset("race", "all")
        df_race_train = race["train"].to_pandas()
        df_race_val = race["validation"].to_pandas()
        df_race_test = race["test"].to_pandas()

        df_race_train = convert_dataframe_race(df_race_train)
        df_race_val = convert_dataframe_race(df_race_val)
        df_race_test = convert_dataframe_race(df_race_test)

        quail = datasets.load_dataset("quail", None)
        df_race_train = race["train"].to_pandas()
        df_race_val = race["validation"].to_pandas()
        df_race_test = race["test"].to_pandas()

        df_race_train = convert_dataframe_race(df_race_train)
        df_race_val = convert_dataframe_race(df_race_val)
        df_race_test = convert_dataframe_race(df_race_test)

        # DREAM
        # MCTEST
        # COSMOSQA


    # TODO: create binary megadaset
    # quail = datasets.load_dataset("quail", None)
    # df_quail_train = quail["train"].to_pandas()
    # df_quail_val = quail["validation"].to_pandas()
    # df_quail_test = quail["challenge"].to_pandas()

    # df_quail_train = convert_dataframe_quail(df_quail_train)
    # df_quail_val = convert_dataframe_quail(df_quail_val)
    # df_quail_test = convert_dataframe_quail(df_quail_test)

    # race = datasets.load_dataset("race", "all")
    # df_race_train = race["train"].to_pandas()
    # df_race_val = race["validation"].to_pandas()
    # df_race_test = race["test"].to_pandas()

    # df_race_train = convert_dataframe_race(df_race_train)
    # df_race_val = convert_dataframe_race(df_race_val)
    # df_race_test = convert_dataframe_race(df_race_test)

    # frames = [df_quail_train, df_race_train]
    # result_train = pd.concat(frames)

    # frames = [df_quail_val, df_race_val]
    # result_val = pd.concat(frames)

    # frames = [df_quail_test, df_race_test]
    # result_test = pd.concat(frames)

    # result_train.to_csv("./data/mega_train.csv", index=False)
    # result_val.to_csv("./data/mega_val.csv", index=False)
    # result_test.to_csv("./data/mega_test.csv", index=False)

    # TODO: create multi megadaset
