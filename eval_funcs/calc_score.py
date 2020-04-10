import numpy as np
import pandas as pd
from preprocessing_data import feature_select, preprocessing
def calc_score(scores_with_data):
    list_scores = []
    for score in scores_with_data:
        list_scores.append(float(score.split()[1]))
    print(len(list_scores))
    return np.array(list_scores).mean()



if __name__ == "__main__":
    df = pd.read_csv('datasets/regression/automobile.data')
    pp = preprocessing.PreProcessing(df, -1)
    pp.processing_missing_values()
    pp.one_hot_encoder_categorical_features()
    df = pp.get_dataframe()
    target = np.array(df[df.columns[-1]]).mean()

    print(target)
    # with open('results/movement_libras_stats2.txt', 'r') as f:
    #     scores = f.read().splitlines()
    #     new_scores = []
    #     for score in scores:
    #         if score == '':
    #             break
    #         new_scores.append(score)
    #     # scores = scores[:scores.find('')]
    #     scores = new_scores
    #     print(scores)
    #     mymean = calc_score(scores)
    #     # tpotmean = calc_score(scores[1::2])
    # # df = pd.read_csv('datasets/fundamentals.csv')
    # # df = df.dropna()
    # # var = np.average(df['Estimated Shares Outstanding'])
    # # print(var)
    # print(mymean)
    # # print(tpotmean)
    #
    #
    #
