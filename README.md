# EC-code

To reproduce our expriments, please refer to the following steps:

1. Install libraries.

	- [EGG Toolkit](https://github.com/facebookresearch/EGG).

	- Python packages:

		``` 
		numpy==1.23.3
		scipy==1.9.1
		torch==2.0.0
		torchvision==0.15.1
		tqdm==4.64.1
		```

2. Generating data. More details in `rule-RAVEN/README.md`

3. Pre-training the speaker. More details in `EC-stage-1/README.md`

4. Joint-training the speaker and the listener and dump the messages. More details in `EC-stage-2/README.md`

5. Transferring across different attribute values. More details in `EC-ETL/README.md`