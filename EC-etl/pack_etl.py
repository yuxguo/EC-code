import os
import shutil

def pack_data(pack_root, data_root):
    dst_path = os.path.join(pack_root, data_root)
    os.makedirs(dst_path, exist_ok=True)
    for d in os.listdir(data_root):
        src_path = os.path.join(data_root, d)
        os.system("cp -r %s %s" % (src_path, dst_path))
    

def pack_dump(pack_root, dump_root):
    # log and message
    dst_path = os.path.join(pack_root, dump_root)
    os.makedirs(dst_path, exist_ok=True)
    # group by exp, agg seed
    files = os.listdir(dump_root)
    exps = set(
        [
           k.split("_seed_")[0] + "_" + "_".join(k.split("_seed_")[1].split("_")[1:])
           for k in files
        ]
    ) # without warmup

    for e in exps:
        exp_dst_dir = os.path.join(dst_path, e)
        os.makedirs(exp_dst_dir, exist_ok=True)

        total_seeds = [
            "%s_seed_%d_%s" % ("_".join(e.split("_")[0:7]), i, "_".join(e.split("_")[7:])) if "random_agent" in e else "%s_seed_%d_%s" % ("_".join(e.split("_")[0:6]), i, "_".join(e.split("_")[6:])) 
            for i in range(8)
        ]
        # copy logs
        exp_log_dst_dir = os.path.join(exp_dst_dir, "logs")
        exp_acc_dst_dir = os.path.join(exp_dst_dir, "accs")
        os.makedirs(exp_log_dst_dir, exist_ok=True)
        os.makedirs(exp_acc_dst_dir, exist_ok=True)
        for seed in total_seeds:
            src_log = os.path.join(dump_root, seed, "log.log")
            seed_idx = seed.split("_seed_")[1].split("_")[0]
            dst_log = os.path.join(exp_log_dst_dir, "%s.log" % seed_idx)
            os.system("cp -r %s %s" % (src_log, dst_log))
        
        # copy step accs
        for seed in total_seeds:
            src_acc = os.path.join(dump_root, seed, "step_accs.pkl")
            seed_idx = seed.split("_seed_")[1].split("_")[0]
            dst_acc = os.path.join(exp_acc_dst_dir, "%s.pkl" % seed_idx)
            os.system("cp -r %s %s" % (src_acc, dst_acc))

def main():
    pack_root = "./dump_etl_pack"
    pack_zip = "dump_etl_pack.zip"
    data_root = "./data/paper"
    dump_root = "./dump_paper"
    os.makedirs(pack_root, exist_ok=True)
    os.system("rm -rf %s/*" % pack_root)
    os.system("rm -rf %s" % pack_zip)

    # pack_data(pack_root, data_root)
    pack_dump(pack_root, dump_root)

    os.system("zip -vr %s %s" % (pack_zip, pack_root))


if __name__ == "__main__":
    main()

