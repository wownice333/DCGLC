import argparse

def arg_parse():
        parser = argparse.ArgumentParser(description='DCGCL Arguments.')
        parser.add_argument('--DS', dest='DS', help='Dataset', default='BZR')
        parser.add_argument('--local', dest='local', action='store_const', 
                const=True, default=False)
        parser.add_argument('--glob', dest='glob', action='store_const', 
                const=True, default=False)
        parser.add_argument('--prior', dest='prior', action='store_const', 
                const=True, default=False)
        parser.add_argument('--lr', dest='lr', type=float,
                help='Learning rate.',default=0.001)
        parser.add_argument('--num-gc-layers', dest='num_gc_layers', type=int, default=4,
                help='Number of graph convolution layers before each pooling')
        parser.add_argument('--pretrain_epoch', dest='pretrain_epoch', type=int, help='Pre-training Epochs', default=100)
        parser.add_argument('--hidden-dim', dest='hidden_dim', type=int, default=16,
                help='')
        parser.add_argument('--cluster_emb', dest='cluster_emb', type=int, default=10, help='cluster layer embedding dimension')
        parser.add_argument('--d', dest='d', type=int, default=5, help='')
        parser.add_argument('--eta', dest='eta', type=int, default=2, help='')
        parser.add_argument('--clusters', dest='clusters', type=int, default=2, help='')    
        parser.add_argument('--preprocess', dest='preprocess', default=False) 
        parser.add_argument('--loss', dest='loss', default='kl')
        parser.add_argument('--aug', type=str, default='minmax')
        parser.add_argument('--gamma', type=float, default=0.01)
        parser.add_argument('--lamda', type=float, default=1)
        parser.add_argument('--eval', type=bool, default=True, help='whether load saved weight files or not')
        parser.add_argument('--beta', type=float, default=10)
        parser.add_argument('--mode', type=str, default='fast')
        parser.add_argument('--seed', type=int, default=0)


        return parser.parse_args()

