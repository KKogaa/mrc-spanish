import pandas as pd
import regex as re


def remove_first(texts):
    texts[1] = texts[1][1:]
    texts[2] = texts[2][1:]
    texts[3] = texts[3][1:]
    return texts


if __name__ == "__main__":

    df_1_train = pd.read_csv("data/translated_race_train.csv")
    df_1_val = pd.read_csv("data/translated_race_val.csv")
    df_1_test = pd.read_csv("data/translated_race_test.csv")

    df_2_train = pd.read_csv("data/translated_race_train_2.csv")
    df_2_train["translated_answers"] = df_2_train["translated_answers"].str.split(",")
    df_2_val = pd.read_csv("data/translated_race_val_2.csv")
    df_2_val["translated_answers"] = df_2_val["translated_answers"].str.split(",")
    df_2_test = pd.read_csv("data/translated_race_test_2.csv")
    df_2_test["translated_answers"] = df_2_test["translated_answers"].str.split(",")

    df_3_train = pd.read_csv("data/translated_race_train_3.csv")
    df_3_val = pd.read_csv("data/translated_race_val_3.csv")
    df_3_test = pd.read_csv("data/translated_race_test_3.csv")

    df_1_train = pd.merge(
        df_1_train, df_2_train["translated_answers"], left_index=True, right_index=True
    )
    df_1_val = pd.merge(
        df_1_val, df_2_val["translated_answers"], left_index=True, right_index=True
    )
    df_1_test = pd.merge(
        df_1_test, df_2_test["translated_answers"], left_index=True, right_index=True
    )

    df_1_train = pd.merge(
        df_1_train,
        df_3_train[["example_id", "translated_context"]],
        how="inner",
        on="example_id",
    )
    df_1_val = pd.merge(
        df_1_val,
        df_3_val[["example_id", "translated_context"]],
        how="inner",
        on="example_id",
    )
    df_1_test = pd.merge(
        df_1_test,
        df_3_test[["example_id", "translated_context"]],
        how="inner",
        on="example_id",
    )

    df_1_train["translated_answers"] = df_1_train["translated_answers"].map(
        lambda x: ",".join(map(str, x))
    )
    df_1_val["translated_answers"] = df_1_val["translated_answers"].map(
        lambda x: ",".join(map(str, x))
    )
    df_1_test["translated_answers"] = df_1_test["translated_answers"].map(
        lambda x: ",".join(map(str, x))
    )

    df_1_train.to_csv("data/final_translated_race_train.csv", index=False)
    df_1_val.to_csv("data/final_translated_race_val.csv", index=False)
    df_1_test.to_csv("data/final_translated_race_test.csv", index=False)

    # df_1_train = pd.read_csv("data/final_translated_race_train.csv")
    # df_1_train['translated_answers'] = df_1_train['translated_answers'].str.split(',')
    # df_1_train['translated_answers'] = df_1_train['translated_answers'].map(lambda x: ','.join(map(str, x)))
    # df_1_val = pd.read_csv("data/final_translated_race_val.csv")
    # df_1_test = pd.read_csv("data/final_translated_race_test.csv")

    # df_1_train = pd.read_csv("data/translated_quail_train.csv")
    # df_1_train['translated_answers'] = df_1_train['translated_answers'].str.split(',')
    # print(len(df_2_train.iloc[0]["translated_answers"]))
    # for ans in df_2_train.iloc[0]["translated_answers"]:
    #     print(ans)

    # df_2_train['translated_answers'] = df_2_train['translated_answers'].str.translate(str.maketrans({"'":None}))
    # df_2_train['translated_answers'] = df_2_train['translated_answers'].str.replace('[', '')
    # df_2_train['translated_answers'] = df_2_train['translated_answers'].str.replace(']', '')
    # df_2_train['translated_answers'] = df_2_train['translated_answers'].str.split(',')
    # df_2_train['translated_answers'] = df_2_train.apply(lambda x: remove_first(x.translated_answers), axis=1)
    # df_2_train['translated_answers'] = df_2_train['translated_answers'].map(lambda x: ','.join(map(str, x)))
    # df_2_train.to_csv("data/translated_race_train_2.csv")

    # df_2_val['translated_answers'] = df_2_val['translated_answers'].str.translate(str.maketrans({"'":None}))
    # df_2_val['translated_answers'] = df_2_val['translated_answers'].str.replace('[', '')
    # df_2_val['translated_answers'] = df_2_val['translated_answers'].str.replace(']', '')
    # df_2_val['translated_answers'] = df_2_val['translated_answers'].str.split(',')
    # df_2_val['translated_answers'] = df_2_val.apply(lambda x: remove_first(x.translated_answers), axis=1)
    # df_2_val['translated_answers'] = df_2_val['translated_answers'].map(lambda x: ','.join(map(str, x)))
    # df_2_val.to_csv("data/translated_race_val_2.csv")
    # print(df_2_train.iloc[0]["translated_answers"])

    # df_2_train['translated_answers'] = df_2_train['translated_answers'].map(lambda x: ','.join(map(str, x)))
