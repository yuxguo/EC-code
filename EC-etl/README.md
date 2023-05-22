# EC ETL

To reproduce our expriments, please refer to the following steps:

1. Transferring dumped message.

	- `cd build_message`

	- Set generated data paths and dumped message paths in `[agent, random_agent, rule]_message.py`.
	- `python [agent, random_agent, rule]_message.py` to generate mapped messages in `[agent, random_agent, rule]_message/`

2. Prepare data and pre-trained speaker. Set generated data path in `EC/args.py: data_dir` and pre-trained speaker model path in `EC/args.py: speaker_pretrain_path`.

3. Training the listener with the mapped message and data. 

	`bash run.sh <start_seed> <end_seed> <message_mode: [agent, random_agent, rule]> <source N: [20, 30, 40]> <target N: [20, 30, 40, 80]>`

	Note that when `message_mode = "rule"`, `source N` must equal to `target N`.

	

	Example 1: `bash run.sh 0 8 agent 20 40`

	Meaning 1: Train seed `[0, 1, 2, 3, 4, 5, 6, 7]`, on `agent` message, from 20 to 40.

	Example 2: `bash run.sh 0 8 rule 30 30`

	Meaning 2: Train seed `[0, 1, 2, 3, 4, 5, 6, 7]`, on `rule(ideal)` message, with `N=30`.



Other utilities: `pack*.py` is for extracting and packing the results to a `.zip` file.