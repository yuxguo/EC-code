import pickle
import os
import numpy as np
import copy

'''
extract rule from data as message
'''
message_root = '../../EC-stage-2/dump_paper/4x15_l2_inpo_%d_seed_%d/message/gen0/%s_message.pkl' # vr, seed
data_root = '../../EC-stage-2/data/paper/l2_inpo_%d/%s_visual.pkl'
output_root = "./agent_message/4x15_l2_inpo_%d_seed_%d_to_%d/" # src_vr seed to dst_vr 
rules_convert_dict = {
    "Constant": {
        "0": 1
    },
    "Progression": {
        "-2": 2,
        "-1": 3,
        "1": 4,
        "2": 5
    },
    "Arithmetic": {
        "-1": 6,
        "1": 7
    },
    "Comparison": {
        "-1": 8,
        "1": 9
    },
    "Varprogression": {
        "1": 10,
        "2": 11,
        "-1": 12,
        "-2": 13,
    },
}

def load_message(msg_path):
    with open(msg_path, "rb") as f:
        data = pickle.load(f)
    result = []
    for d in data:
        m = d['message'].squeeze()[:-1]
        result.append(m)
    return result

def load_rule(data_path):
    # return list of tuples for hash
    with open(data_path, "rb") as f:
        data = pickle.load(f)
    result = []
    for d in data:
        r = d['rules']
        arr = tuple([rules_convert_dict[i[1].capitalize()][str(i[2])] for i in r])
        result.append(arr)
    return result

def pair(src_message, src_rule, dst_rule):
    '''
    generate dst message
    '''
    # first step, statistic
    src_stat = dict()
    src_count = dict()
    for sm, sr in zip(src_message, src_rule):
        if src_stat.get(sr, None) is None:
            src_stat.update({sr: [sm]})
            src_count.update({sr: 0})
        else:
            src_stat[sr].append(sm)
    
    for k, v in src_stat.items():
        assert len(v) == 10
    # second step, assign
    result = []
    for dr in dst_rule:
        cnt = src_count[dr]
        item = copy.deepcopy(src_stat[dr][cnt])
        src_count[dr] += 1
        result.append(item)

    return result

def main():
    for src_vr in [20, 30, 40]:
        for dst_vr in [20, 30, 40, 80]:
            if src_vr == dst_vr:
                continue
            for S in range(8):
                print("Trained on value range %d seed %d, test on value range %d." % (src_vr, S, dst_vr))
                # train + test in src
                src_message, src_rule = [], []
                for mode in ['train', 'test']:
                    message_mode = "valid" if mode == "validation" else mode
                    src_message_path = message_root % (src_vr, S, message_mode)
                    src_data_path = data_root % (src_vr, mode)
                    src_message += load_message(src_message_path)
                    src_rule += load_rule(src_data_path)

                for mode in ['train', 'validation', 'test']:
                    dst_data_path = data_root % (dst_vr, mode)
                    dst_rule = load_rule(dst_data_path)
                    dst_message = pair(src_message, src_rule, dst_rule)
                    output_dir = output_root % (src_vr, S, dst_vr)
                    os.makedirs(output_dir, exist_ok=True)
                    
                    output_path = os.path.join(output_dir, "%s_external_message_%s.pkl" % (mode, "agent"))
                    with open(output_path, "wb") as f:
                        pickle.dump(dst_message, f)

if __name__ == '__main__':
    main()