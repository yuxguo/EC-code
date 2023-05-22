# EC stage 1

To reproduce our expriments, please refer to the following two steps:

1. Prepare data for pretrain. Set generated data path in `EC/args.py: data_dir`.
2. Set attribute value $N$ in `EC/args.py: symbol_onehot_dim` and pre-train the speaker `python main.py`.
