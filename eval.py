import pandas as pd
import numpy as np


class Topk:
    def __init__(self, K, test_df):
        self.K = K
        self.topk = K
        self.test_df = test_df

    def TOPK(self, R, T):
        res = {}
        p = 4
        res['PRECISION@' + str(self.K)] = round(self.PRECISION(R, T), p)
        res['RECALL@' + str(self.K)] = round(self.RECALL(R, T), p)
        res['HR@' + str(self.K)] = round(self.HR(R, T), p)
        res['MAP@' + str(self.K)] = round(self.MAP(R, T), p)
        res['MRR@' + str(self.K)] = round(self.MRR(R, T), p)
        res['NDCG@' + str(self.K)] = round(self.NDCG(R, T), p)
        res['AUTC@' + str(self.K)] = round(self.AUTC(), p)
        return res

    def evaluate(self):
        temp = self.test_df[['userid', 'feedid', 'play', 'test_label', 'pred']]
        true_df = temp[temp.test_label == 1].groupby(['userid']).agg({'feedid': lambda x: \
            list(x)}).reset_index().sort_values(by=['userid'])
        x = pd.DataFrame(temp[~temp.userid.isin(true_df['userid'])]['userid'].drop_duplicates())
        x['feedid'] = [[] for i in range(len(x))]
        #         true_df = pd.concat([true_df,x])
        temp = temp[temp.userid.isin(true_df['userid'])]
        temp = temp.sort_values(by=['userid', 'pred'], ascending=False)
        rank_df = temp.groupby(['userid']).agg({'feedid': lambda x: list(x)}).reset_index().sort_values(by=['userid'])
        rank_df['top' + str(self.topk)] = rank_df['feedid'].apply(lambda x: x[:self.topk] if len(x) >= self.topk else x)

        assert len(true_df) == len(rank_df)

        df = pd.merge(left=true_df, right=rank_df[['userid', 'top' + str(self.topk)]], on=['userid'])

        assert len(df) == len(df.dropna(how='any'))

        T = df['feedid'].tolist()
        R = df['top' + str(self.topk)].tolist()

        res = self.TOPK(R, T)
        print('RECALL@{:d} {:.4f} | MAP@{:d} {:.4f} | NDCG@{:d} {:.4f} | AUTC@{:d} {:.4f}'
              .format(self.topk, res['RECALL@' + str(self.topk)],
                      self.topk, res['MAP@' + str(self.topk)],
                      self.topk, res['NDCG@' + str(self.topk)],
                      self.topk, res['AUTC@' + str(self.topk)]))

        self.summary(self.test_df)

    def PRECISION(self, R, T):
        assert len(R) == len(T)
        res = 0
        for i in range(len(R)):
            res += len(set(R[i]) & set(T[i])) / len(R[0])
        return res / len(R)

    def RECALL(self, R, T):
        assert len(R) == len(T)
        res = 0
        for i in range(len(R)):
            if len(T[i]) > 0:
                res += len(set(R[i]) & set(T[i])) / len(T[i])
        return res / len(R)

    def HR(self, R, T):
        assert len(R) == len(T)
        up = 0
        down = len(R)
        for i in range(len(R)):
            if len(set(R[i]) & set(T[i])) > 0:
                up += 1
        return up / down

    def MAP(self, R, T):
        assert len(R) == len(T)
        up = 0
        down = len(R)
        for i in range(len(R)):
            temp = 0
            hit = 0
            for j in range(len(R[i])):
                if R[i][j] in T[i]:
                    hit += 1
                    temp += hit / (j + 1)
            if hit > 0:
                up += temp / len(T[i])
        return up / down

    def MRR(self, R, T):
        assert len(R) == len(T)
        up = 0
        down = len(R)
        for i in range(len(R)):
            index = -1
            for j in range(len(R[i])):
                if R[i][j] in T[i]:
                    index = R[i].index(R[i][j])
                    break
            if index != -1:
                up += 1 / (index + 1)
        return up / down

    def dcg(self, hits):
        res = 0
        for i in range(len(hits)):
            res += (hits[i] / np.log2(i + 2))
        return res

    def NDCG(self, R, T):
        assert len(R) == len(T)
        up = 0
        down = len(R)
        for i in range(len(R)):
            hits = []
            for j in range(len(R[i])):
                if R[i][j] in T[i]:
                    hits += [1.0]
                else:
                    hits += [0.0]
            if sum(hits) > 0:
                up += (self.dcg(hits) / (self.dcg([1.0 for i in range(len(T[i]))]) + 1))  # 来自wiki的定义，idcg应该是对目标排序。
        return up / down

    def AUTC(self):
        t = self.test_df.groupby(['userid']).apply(lambda x: self.play_time(x, self.topk))
        return t.sum() / len(t)

    def play_time(self, x, topk):
        temp = x.sort_values(by=['pred'], ascending=False)[:topk]
        rank_time = temp['play'].sum()
        sum_time = x['play'].sum()
        return rank_time / sum_time

    def summary(self, test):
        # test.groupby(['duration_level'])['pred'].mean().tolist()
        print(test.groupby(['duration_level'])['pred'].mean())

        temp = test.sort_values(by=['userid', 'pred'], ascending=False)
        maps = test[['feedid', 'duration_level']].drop_duplicates(['feedid'])
        print(maps.groupby(['duration_level'])['feedid'].count())

        topk = 5
        rank_df = temp.groupby(['userid']).agg({'feedid': lambda x: list(x)}).reset_index().sort_values(by=['userid'])
        rank_df['top' + str(topk)] = rank_df['feedid'].apply(
            lambda x: maps[maps.feedid.isin(x[:topk])]['duration_level'].tolist() if len(x) >= topk else x)

        x = rank_df['top5'].tolist()

        temp = []
        for i in x:
            temp += i
        fre = []
        for i in range(6):
            fre += [temp.count(i)]
        print(fre)