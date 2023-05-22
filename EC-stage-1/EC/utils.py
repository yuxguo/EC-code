import torch
import os


def save_checkpoint(epoch, best_validation_acc, model, optimizer, ckpt_dir, filename):
    os.makedirs(ckpt_dir, exist_ok=True)
    torch.save({
        "epoch": epoch,
        "best_validation_acc": best_validation_acc,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict()
    }, os.path.join(ckpt_dir, filename))

def load_checkpoint(ckpt_dir, filename):
    checkpoint = torch.load(os.path.join(ckpt_dir, filename))
    return checkpoint['epoch'], checkpoint['best_validation_acc'], checkpoint['model'], checkpoint['optimizer']
