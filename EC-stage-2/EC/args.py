import os


class Args(object):
    def __init__(self, seed=0, gener_level="l1_inpo", dump_message=False, symbol_onehot_dim=20):
        
        self.seed = seed
        self.gener_level = gener_level # [l1_inpo, l1_expo, l2_inpo, l2_expo]
        self.symbol_onehot_dim = symbol_onehot_dim

        # CUDA part

        # self.cuda_visible_devices = "0, 1, 2, 3"
        
        self.cuda_visible_devices = "%d" % (self.seed % 8)
        # self.cuda_visible_devices = "2, 3"


        # visual & symbol part

        self.use_resnet = False

        self.speaker_freeze_visual = False
        self.listener_freeze_visual = False

        self.speaker_freeze = False
        self.listener_freeze = False

        self.speaker_use_pretrain_model = True
        self.speaker_pretrain_path = "./dump_paper/4x15_warmup_88_%d/checkpoints/best_epoch.pth" % (self.symbol_onehot_dim)
        self.speaker_pretrain_params_key = "speaker"

        self.listener_use_pretrain_model = False
        self.listener_pretrain_path = ""
        self.listener_pretrain_params_key = "listener"
        
        # hyper-paramerter part
        self.max_pooling_message_embedding = False
        self.mlp_pooling_message_embedding = False

        assert not(self.max_pooling_message_embedding and self.mlp_pooling_message_embedding)


        self.image_embedding_dim = 4 * self.symbol_onehot_dim # 80, 120, 160
        self.message_embedding_dim = 5 * self.image_embedding_dim if not self.max_pooling_message_embedding and not self.mlp_pooling_message_embedding else 80
        

        self.message_max_len = 4
        self.vocab_size = 15


        self.message_encoder_rnn_hidden_dim = 5 * self.image_embedding_dim if not self.max_pooling_message_embedding and not self.mlp_pooling_message_embedding else 80
        self.message_decoder_rnn_hidden_dim = 5 * self.image_embedding_dim if not self.max_pooling_message_embedding and not self.mlp_pooling_message_embedding else 80

        self.message_encoder_transformer_hidden_dim = 5 * self.image_embedding_dim if not self.max_pooling_message_embedding and not self.mlp_pooling_message_embedding else 80
        self.message_decoder_transformer_hidden_dim = 5 * self.image_embedding_dim if not self.max_pooling_message_embedding and not self.mlp_pooling_message_embedding else 80

        self.cell = 'gru'
        self.length_cost = 0
        self.gumbel_temperature = 1.0
        
        self.message_length_cost = 0

        self.lr = 3e-3
        self.vlr = 5e-3
        self.slr = 1e-4
        self.weight_decay = 0.01
        

        self.rand_perm = False
        self.message_embedding_noise = "none" # gaussian, none

        self.listener_reset = False
        self.listener_reset_times = 50
        self.listener_reset_cycle = 20
        

        self.max_epoches = 200
        # self.max_epoches = 0
        self.dataloader_num_workers = 8
        self.train_batch_size = 512
        # self.train_batch_size = 2048
        self.test_batch_size = 32
        self.validation_batch_size = 32

        # self.data_dir = './data/paper/ablation/%s' % (self.gener_level)
        self.data_dir = './data/paper/%s_%d/' % (self.gener_level, self.symbol_onehot_dim)
        
        self.data_format_str = '%s_visual.pkl'
        
        self.auto_resume = True

        self.agent_type = 'rnn_reinforce' # ["rnn_gs", "rnn_reinforce", "transformer_reinforce"]
        # self.rule = True
        self.rule = False
        self.visual, self.symbol = (False, True)
        # self.visual, self.symbol = (True, False)
        assert self.visual ^ self.symbol
        self.null_message = False
        self.const_message = False
        self.use_message_max_len = False

        self.add_LN = False

        self.use_constrative = False

        self.symbol_attr_dim = 4
        # self.symbol_onehot_dim = 20
        self.symbol_model_hidden_dims = [32,]
        self.rules_dim = 15

        

        self.rnn_gs_speaker_configs = {
            "vocab_size": self.vocab_size, 
            "embed_dim": self.message_embedding_dim, 
            "hidden_size": self.message_encoder_rnn_hidden_dim, 
            "cell": self.cell,
            "max_len": self.message_max_len,
            "temperature": self.gumbel_temperature,
            "trainable_temperature": False,
            "straight_through": False
        }

        self.rnn_gs_listener_configs = {
            "vocab_size": self.vocab_size, 
            "embed_dim": self.message_embedding_dim, 
            "hidden_size": self.message_decoder_rnn_hidden_dim, 
            "cell": self.cell,
        }

        self.rnn_reinforce_speaker_configs = {
            "vocab_size": self.vocab_size, 
            "embed_dim": self.message_embedding_dim, 
            "hidden_size": self.message_encoder_rnn_hidden_dim, 
            "cell": self.cell,
            "max_len": self.message_max_len,
        }

        self.rnn_reinforce_listener_configs = {
            "vocab_size": self.vocab_size, 
            "embed_dim": self.message_embedding_dim, 
            "hidden_size": self.message_decoder_rnn_hidden_dim, 
            "cell": self.cell,
        }

        self.transformer_reinforce_speaker_configs = {
            "vocab_size": self.vocab_size, 
            "embed_dim": self.message_embedding_dim, 
            "max_len": self.message_max_len,
            "num_layers": 1,
            "num_heads": 4,
            "hidden_size": self.message_encoder_transformer_hidden_dim,
            "generate_style": "standard",
            "causal": True,
        }

        self.transformer_reinforce_listener_configs = {
            "vocab_size": self.vocab_size, 
            "max_len": self.message_max_len,
            "embed_dim": self.message_embedding_dim, 
            "num_heads": 4,
            "hidden_size": self.message_decoder_transformer_hidden_dim,
            "num_layers": 1,
            "positional_emb": True,
            "causal": True,
        }
        

        self.visual_shared_mlp_configs = {
            "groups": 10, 
            "group_input_dim": 256, 
            "group_output_dim": 8,
            "hidden_dims": [128], 
            "add_res_block": True, 
            "nr_mlps": 1, 
            "flatten": True,
            "shared": True
        }

        self.discri_analogy_shared_mlp_configs = {
            "groups": self.image_embedding_dim, 
            "group_input_dim": 3, 
            "group_output_dim": 1,
            "hidden_dims": [64, 32], 
            "add_res_block": True, 
            "nr_mlps": 5, 
            "flatten": True,
            "shared": True
        }

        self.recon_analogy_shared_mlp_configs = {
            "groups": self.symbol_attr_dim, # 4
            "group_input_dim": 2 * self.image_embedding_dim // self.symbol_attr_dim, # 40
            "group_output_dim": self.image_embedding_dim // self.symbol_attr_dim, # 20
            "hidden_dims": [64, 32], 
            "add_res_block": True, 
            "nr_mlps": self.rules_dim, # 10 
            "flatten": False,
            "shared": True
        }

        self.constrative_analogy_shared_mlp_configs = {
            "groups": 80, 
            "group_input_dim": 2, 
            "group_output_dim": 1,
            "hidden_dims": [64, 32], 
            "add_res_block": True, 
            "nr_mlps": 5, 
            "flatten": True,
            "shared": True
        }

        # self.execution_id = "test"

        # self.execution_id = '%dx%d_else_88_ood_expo_l2_20_seed_0' % (self.message_max_len, self.vocab_size)
        self.execution_id = '%dx%d_%s_%d_seed_%d' % (self.message_max_len, self.vocab_size, self.gener_level, self.symbol_onehot_dim, self.seed)


        # self.execution_id = 'SCL_egg_%dx%d_symbol_rnn_discri_reinforce_vlenmsg_else_88_inpo_20' % (self.message_max_len, self.vocab_size)
        # self.execution_id = 'SCL_egg_%dx%d_symbol_rnn_discri_gs_vlenmsg_ptlistener_r_cs_10k_num' % (self.message_max_len, self.vocab_size)
        # self.execution_id = 'SCL_egg_%dx%d_%dx%d_symbol_1e41e-2cs_rule_10k' % (self.message_max_len, self.vocab_size, self.listener_reset_times, self.listener_reset_cycle)
        self.dump_root = './EC-new/dump_paper/'
        self.dump_dir = os.path.join(self.dump_root, self.execution_id)

        
        self.dump_message = dump_message
        self.discri_game = True
        
    
    def get_execution_id(self):
        pass
