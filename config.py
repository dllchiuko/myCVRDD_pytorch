import argparse

args = argparse.ArgumentParser()

# general
args.add_argument('--lr', type=float, default=1e-5)
args.add_argument('--batch_size', type=int, default=512)
args.add_argument('--epochs', type=int, default=64)
args.add_argument('--max_degree', type=int, default=3)
args.add_argument('--embedding_dim', type=int, default=64)
args.add_argument('--bias_dim', type=int, default=64)
args.add_argument('--mlp_dims', type=list, default=[300, 200, 100])
args.add_argument('--dropout', type=list, default=[0.2, 0.2, 0.2])
args.add_argument('--alpha', type=float, default=1.0)
args.add_argument('--kl', type=float, default=0.8)
args.add_argument('--fusion_mode', type=str, default='sum')  # 'sum', 'hm', 'mp'

args.add_argument('--seed', type=int, default=40)
args.add_argument('--samples', type=int, default=100000)  # None, int
args.add_argument('--training_label', type=str, default='pcr')  # 'pcr', 'fp'
args.add_argument('--train', type=bool, default=True)  # True, False

args = args.parse_args()

