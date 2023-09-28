import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

from utils import *
from config import *

import warnings
warnings.filterwarnings("ignore")


class Data(object):
    # def __init__(self):
    #     return

    def load_data(self):
        action = pd.read_csv("./data/user_action.csv")
        feed = pd.read_csv("./data/feed_info.csv")
        feed = feed[['feedid', 'authorid', 'videoplayseconds']]
        feed_emb = pd.read_csv("./data/feed_embeddings.csv")

        action = pd.merge(left=action, right=feed, on=['feedid'], how='left').dropna(how='any')
        action = pd.merge(left=action, right=feed_emb, on=['feedid'], how='left').dropna(how='any')

        action = action[action.videoplayseconds <= 60]

        # calculate pcr and duration level
        action['pcr'] = action['play'] / (action['videoplayseconds'] * 1000.0)  # play单位为毫秒，1s = 1000ms
        action['pcr'] = action['pcr'].apply(lambda x: min(1.0, x))

        action['duration_level'] = (action[['videoplayseconds']] - 1) // 10  # [1, 11)为level 0，[11, 21)为level 1

        # calculate threshold
        threshold = get_threshold(action)
        action = pd.merge(left=action, right=threshold, on=['duration_level'], how='left')

        # define labels
        action['train_label'] = (action['pcr'] >= action['threshold']).astype('int32')
        action['finish_playing'] = (action['pcr'] >= 1.0).astype('int32')
        action['test_label'] = (action['read_comment'] | action['like'])  # 按位或，满足其一时值为1，否则为0
        action = action[~((action.train_label == 0) & (action.test_label == 1))]  # 取反？？？？？？不取没有观看但有互动的数据

        # reorder ids
        action['userid'] = reorder_id(action, 'userid')
        action['feedid'] = reorder_id(action, 'feedid')
        action['authorid'] = reorder_id(action, 'authorid')

        # get embeddings of items
        feed_emb = action[['feedid', 'feed_embedding']].drop_duplicates(['feedid']).sort_values(by=['feedid'])  # 唯一一个id对应唯一的embedding
        from sklearn.decomposition import PCA

        pca = PCA(n_components=64)  # 使用主成分分析降维，512 -> 64，降维后返回后的数据不是原始数据
        emb = np.array(feed_emb['feed_embedding'].apply(lambda x: [float(i) for i in x.split(' ')[:-1]]).tolist())
        res = pca.fit_transform(emb)
        emb = pd.DataFrame(res, columns=[i for i in range(64)])

        # save action as csv_file and embeddings as npy
        if not os.path.exists('./input/'):
            os.mkdir('./input/')

        np.save('./input/feedid_emb_64.npy', emb.to_numpy())  # 数字数组数据类型

        action.drop('feed_embedding', axis=1, inplace=True)
        action.to_csv("./input/action.csv", index=False)

    def generate_data(self, batch_size):
        if args.samples is not None:
            action = pd.read_csv('./input/action.csv').iloc[:args.samples, :]
        else:
            action = pd.read_csv('./input/action.csv').iloc[:, :]
            print(len(action))
        action = action.sort_values(by='date_')
        num_features = len(action)
        setattr(Data, 'num_features', num_features)

        feed_emb = torch.from_numpy(np.load('./input/feedid_emb_64.npy'))

        features = ['userid', 'feedid', 'duration_level', 'device', 'authorid']

        features_sizes = []
        features_sizes.extend(int(max(action[i]) + 1) for i in features)
        setattr(Data, 'features_sizes', features_sizes)


        # 按用户划分数据集
        train = action.groupby('userid').apply(lambda x: x[:int(len(x) * 0.6)]).reset_index(drop=True)  # 每个用户有60%划分为训练集
        valid = action.groupby('userid').apply(lambda x: x[int(len(x) * 0.6): int(len(x) * 0.8)]).reset_index(drop=True)
        test = action.groupby('userid').apply(lambda x: x[int(len(x) * 0.8):]).reset_index(drop=True)

        features = ['userid', 'feedid', 'duration_level', 'device', 'authorid']

        def get_inputs(df, is_test=True):
            X = []
            for f in features:
                # X = df[features]
                X.append(torch.from_numpy(df[f].values).unsqueeze(dim=0))
            X = torch.concatenate(X).T

            if is_test:
                y = torch.from_numpy(df['test_label'].values).float()
            else:
                if args.training_label == 'pcr':
                    y = torch.from_numpy(df['train_label'].values).float()
                else:
                    y = torch.from_numpy(df['finish_playing'].values).float()
            return X, y

        def get_loader(inputs):
            x, y = inputs
            return DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=True)

        train_loader = get_loader(get_inputs(train, is_test=False))  # [tensor(), tensor(), ...]
        valid_loader = get_loader(get_inputs(valid, is_test=False))
        test_loader = get_loader(get_inputs(test, is_test=True))

        return train_loader, valid_loader, test_loader, feed_emb, test





