import pickle
import os
import numpy as np

'''
extract rule from data as message
'''
data_root = '../../EC-stage-2/EC-new/data/paper/l2_inpo_%d/'
output_root = "./rule_message/l2_inpo_%d/"
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

def main():
    for vr in [20, 30, 40, 80]:
        data_path = data_root % vr
        output_path = output_root % vr
        for mode in ['train', 'validation', 'test']:
            with open(os.path.join(data_path, "%s_visual.pkl" % mode), "rb") as f:
                data = pickle.load(f)
            result = []
            for d in data:
                r = d['rules']
                arr = np.array([rules_convert_dict[i[1].capitalize()][str(i[2])] for i in r])
                result.append(arr)
            os.makedirs(output_path, exist_ok=True)
            with open(os.path.join(output_path, "%s_external_message_%s.pkl" % (mode, "rule")), "wb") as f:
                pickle.dump(result, f)

if __name__ == '__main__':
    main()