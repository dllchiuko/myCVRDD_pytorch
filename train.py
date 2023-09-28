import torch
import torch.nn as nn
import time
import numpy as np
import random
from sklearn.metrics import auc, roc_curve
from tqdm import tqdm

from data import *
from models import *
from config import *
from eval import *


# set seed
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)


# load data
train_loader, valid_loader, test_loader, feed_embed, test_df = Data().generate_data(args.batch_size)
num_features = getattr(Data(), 'num_features')


# initialize model
save_name = 'myCVRDD_'+args.training_label+str(args.samples)+'.pth' if args.samples is not None else 'myCVRDD_'+args.training_label+'.pth'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device:', device)

model = CVRDD(num_features, args.embedding_dim, args.bias_dim, args.mlp_dims, args.dropout, args.alpha, args.kl, args.fusion_mode)
model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

kl_loss = KLLoss(args.kl)
bce_loss_base = nn.BCELoss()
bce_loss_bias = nn.BCELoss()
mse_loss = nn.MSELoss()


# train the model
def train_and_test(train_loader, valid_loader, test_loader, test_df, training=True):
    # if training:
    global loss, bce_loss_base_, bce_loss_bias_, kl_loss_, auc_score
    min_loss = 2.0

    if args.train:
        model.train()
        print('Start training the CVRDD model...')
        start_time = time.time()

        for epoch in tqdm(range(args.epochs)):
            for idx, (x, y) in enumerate(train_loader):
                ym0d, ym1d, y_pred = model.forward((x, feed_embed))

                # pred = nn.Sigmoid()(ym0d - ym1d)
                # fpr, tpr, thresholds = roc_curve(y.numpy(), pred.detach().numpy(), pos_label=2)
                # auc_score = auc(fpr, tpr)

                kl_loss_ = args.kl * kl_loss((ym0d, ym1d))
                bce_loss_base_ = bce_loss_base(ym0d, y)
                bce_loss_bias_ = bce_loss_bias(ym1d, y)

                bce_loss_ = bce_loss_base_ + args.alpha * bce_loss_bias_
                loss = bce_loss_ + kl_loss_
                # loss = bce_loss_base_
                # loss = mse_loss(y_pred, y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if loss < min_loss:
                    min_loss = loss
                    torch.save(model.state_dict(), save_name)

            if (epoch + 1) % 5 == 0 or (epoch + 1) == args.epochs:
                print(
                    'epoch {:04d}/{:04d} | total loss {:.4f} | bce_base loss {:.4f} | bce_bias loss {:.4f} | kl loss {:.4f} | time {:.4f} '
                    .format(epoch + 1, args.epochs, loss, bce_loss_base_, bce_loss_bias_, kl_loss_, time.time() - start_time))

    if not args.train:
        model.load_state_dict(torch.load(save_name))

    model.eval()
    base = []
    bias = []
    pred = []
    for idx, (x, y) in enumerate(test_loader):
        ym0d, ym1d = model.forward((x, feed_embed))
        test_pred = nn.Sigmoid()(ym0d - ym1d)

        base.extend(ym0d.tolist())
        bias.extend(ym1d.tolist())
        pred.extend(test_pred.tolist())

    test_df['base'] = base
    test_df['bias'] = bias
    test_df['pred'] = pred

    Topk(3, test_df).evaluate()
    Topk(5, test_df).evaluate()


train_and_test(train_loader, valid_loader, test_loader, test_df)



