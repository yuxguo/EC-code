# EC stage 1

This folder contains code for pre-training the speaker. To reproduce our expriments, please refer to the following two steps:

1. Prepare data for pretrain. Set generated data path in `EC/args.py: data_dir`.
2. Create a folder for saving the logs and checkpoints (Defaults `./dump_paper/` in `EC/args.py: dump_dir`).
3. Set attribute value $N \in \{20, 30, 40\}$ in `EC/args.py: symbol_onehot_dim` and pre-train the speaker `python main.py`. You should manually terminate the pre-training process until the accuracy reaches 0.99.
