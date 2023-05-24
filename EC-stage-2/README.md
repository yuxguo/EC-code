# EC stage 2

To reproduce our expriments, please refer to the following two steps:

1. Prepare data and pre-trained speaker model. Set generated data path in `EC/args.py: data_dir` and pre-trained speaker model path in `EC/args.py: speaker_pretrain_path`.

2. Create a folder for saving the logs and checkpoints (Defaults `./dump_paper/` in `EC/args.py: dump_dir`).

3. Joint-training the speaker and the listener. 

  `bash run.sh <start_seed> <end_seed> <[l1_inpo, l1_expo, l2_inpo, l2_expo]> <[20, 30, 40]> <dump_message: [0, 1]>`

  Example 1: `bash run.sh 0 8 l2_inpo 20 0`

  Meaning 1: Train seed `[0, 1, 2, 3, 4, 5, 6, 7]`, on `l2_inpo_20` dataset split, without dump message.

  Example 2: `bash run.sh 0 8 l2_inpo 20 1`

  Meaning 2: Dump message of seed `[0, 1, 2, 3, 4, 5, 6, 7]`, on `l2_inpo_20` dataset split.

Other utilities: `pack*.py` is for extracting and packing the results to a `.zip` file.

