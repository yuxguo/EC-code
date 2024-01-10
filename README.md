# EC-code

Code for `Emergent Communication for Rules Reasoning`, Neurips 2023 Poster.
To reproduce our experiments, please refer to the following steps:

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

If you found this work helpful, please consider citing our work.
```
@inproceedings{
guo2023emergent,
title={Emergent Communication for Rules Reasoning},
author={Yuxuan Guo and Yifan Hao and Rui Zhang and Enshuai Zhou and Zidong Du and Xishan Zhang and Xinkai Song and Yuanbo Wen and Yongwei Zhao and Xuehai Zhou and Jiaming Guo and Qi Yi and Shaohui Peng and Di Huang and Ruizhi Chen and Qi Guo and Yunji Chen},
booktitle={Thirty-seventh Conference on Neural Information Processing Systems},
year={2023},
url={https://openreview.net/forum?id=gx20B4ItIw}
}
```
