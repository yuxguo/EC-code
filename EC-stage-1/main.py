import os
from time import get_clock_info
from EC import Args

args = Args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_visible_devices


from EC.model import EC as Model
from EC.dataset import RAVENDataset
from EC.utils import *

import numpy as np

import torch
# torch.manual_seed(19990809)
from torch.utils.data import Dataset, DataLoader
from torch import nn
import torch.nn.functional as F
import logging
from tqdm import tqdm
import argparse
import pickle


torch.backends.cudnn.benchmark = True

def dump_message(cpkt="best_epoch", message_dump_dir=None):
    message_dump_dir = os.path.join(args.dump_dir, "message", cpkt.strip("best_epoch")) if message_dump_dir is None else os.path.join(args.dump_dir, "message", "gen0")
    os.makedirs(message_dump_dir, exist_ok=True)
    epoch, best_validation_acc, params_dict, _ = load_checkpoint(os.path.join(args.dump_dir, 'checkpoints'),  "%s.pth" % cpkt)
    model = get_model()
    model = get_cuda_model(model)
    print("Load model from: %s, epoch: %d, validation acc: %.4f" % (os.path.join(args.dump_dir, 'checkpoints', "%s.pth" % cpkt), epoch, best_validation_acc))
    model.load_state_dict(params_dict)
    
    model.eval()

    train_set = RAVENDataset('train', args)
    test_set = RAVENDataset('test', args)
    validation_set = RAVENDataset('validation', args)
    train_loader = DataLoader(train_set, batch_size=1, shuffle=False, num_workers=args.dataloader_num_workers)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=args.dataloader_num_workers)
    validation_loader = DataLoader(validation_set, batch_size=1, shuffle=False, num_workers=args.dataloader_num_workers)
    
    train_message = []
    with torch.no_grad():
        with tqdm(total=len(train_loader)) as pbar:
            pbar.set_description("Dump train_set message")
            for _, input_dict in enumerate(train_loader):
                if args.visual:
                    input_dict['image'] = input_dict['image'].cuda()
                elif args.symbol:
                    input_dict['symbol'] = input_dict['symbol'].cuda()
                if args.rule:
                    input_dict['rules'] = input_dict['rules'].cuda()
                input_dict['label'] = input_dict['label'].cuda()

                output_dict = model(input_dict)
                message = output_dict['message'].detach().cpu().numpy()
                train_message.append({"message": message, "pred": output_dict["pred"]})
                pbar.update()

    validation_message = []
    with torch.no_grad():
        with tqdm(total=len(validation_loader)) as pbar:
            pbar.set_description("Dump valid_set message")
            for _, input_dict in enumerate(validation_loader):
                if args.visual:
                    input_dict['image'] = input_dict['image'].cuda()
                elif args.symbol:
                    input_dict['symbol'] = input_dict['symbol'].cuda()
                if args.rule:
                    input_dict['rules'] = input_dict['rules'].cuda()
                input_dict['label'] = input_dict['label'].cuda()

                output_dict = model(input_dict)
                message = output_dict['message'].detach().cpu().numpy()
                validation_message.append({"message": message, "pred": output_dict["pred"]})
                pbar.update()
    
    test_message = []
    with torch.no_grad():
        with tqdm(total=len(test_loader)) as pbar:
            pbar.set_description("Dump test_set message")
            for _, input_dict in enumerate(test_loader):
                if args.visual:
                    input_dict['image'] = input_dict['image'].cuda()
                elif args.symbol:
                    input_dict['symbol'] = input_dict['symbol'].cuda()
                if args.rule:
                    input_dict['rules'] = input_dict['rules'].cuda()
                input_dict['label'] = input_dict['label'].cuda()

                output_dict = model(input_dict)
                message = output_dict['message'].detach().cpu().numpy()
                test_message.append({"message": message, "pred": output_dict["pred"]})
                pbar.update()
    
    print("Dump to pickle file")
    with open(os.path.join(message_dump_dir, "train_message.pkl"), "wb") as f:
        pickle.dump(train_message, f)
    
    with open(os.path.join(message_dump_dir, "valid_message.pkl"), "wb") as f:
        pickle.dump(validation_message, f)
    
    with open(os.path.join(message_dump_dir, "test_message.pkl"), "wb") as f:
        pickle.dump(test_message, f)

def main():
    os.makedirs(args.dump_dir, exist_ok=True)
    best_validation_acc = 0
    logger = get_logger()
    train_loader, test_loader, validation_loader = get_dataloader()
    model = get_model()
    parameter_groups = get_parameter_groups(model)
    optimizer = torch.optim.AdamW(parameter_groups, lr=args.lr, weight_decay=args.weight_decay)
    
    model = get_cuda_model(model)
    
    start_epoch = 0
    best_validation_acc = 0

    ###### bug
    if args.speaker_use_pretrain_model:
        pretrained_dict = torch.load(args.speaker_pretrain_path)['model']
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k.startswith(args.speaker_pretrain_params_key)}
        print(len(pretrained_dict))
        model_dict = model.state_dict()
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    
    if args.listener_use_pretrain_model:
        pretrained_dict = torch.load(args.listener_pretrain_path)['model']
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k.startswith(args.listener_pretrain_params_key)}
        print(len(pretrained_dict))
        model_dict = model.state_dict()
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    ###### end bug

    if args.auto_resume and os.path.exists(os.path.join(args.dump_dir, 'checkpoints', "last_epoch.pth")):
        resume_epoch, resume_best_validation_acc, resume_model_state, resume_optimizer_state =\
            load_checkpoint(os.path.join(args.dump_dir, 'checkpoints'), "last_epoch.pth")
        start_epoch = resume_epoch + 1
        best_validation_acc = resume_best_validation_acc
        print("Auto resume at epoch %d" % start_epoch)
        model.load_state_dict(resume_model_state)
        optimizer.load_state_dict(resume_optimizer_state)

    total_epoches = args.max_epoches if not args.listener_reset else (args.max_epoches + args.listener_reset_times * args.listener_reset_cycle)
    for epoch in range(start_epoch, total_epoches):
        if epoch < args.max_epoches:
            mean_train_acc, mean_train_loss = train(train_loader, optimizer, model, epoch)
            mean_validate_acc, mean_validate_loss = validate(validation_loader, model, epoch)
            mean_test_acc, mean_test_loss = test(test_loader, model, epoch)
            logger.info(
                """
                Epoch %d:
                    mean train acc %.4f, mean train loss: %.4f
                    mean valid acc %.4f, mean valid loss: %.4f
                    mean test acc %.4f, mean test loss: %.4f
                """ % (
                    epoch, 
                    mean_train_acc, mean_train_loss, 
                    mean_validate_acc, mean_validate_loss,
                    mean_test_acc, mean_test_loss
                ))

            save_checkpoint(epoch, best_validation_acc, model, optimizer, os.path.join(args.dump_dir, 'checkpoints'), "last_epoch.pth")
            if mean_validate_acc > best_validation_acc:
                best_validation_acc = mean_validate_acc
                logger.info('Best validation acc: %.4f\n' % best_validation_acc)
                save_checkpoint(epoch, best_validation_acc, model, optimizer, os.path.join(args.dump_dir, 'checkpoints'), "best_epoch.pth")
            if (epoch + 1) % 50 == 0:
                save_checkpoint(epoch, best_validation_acc, model, optimizer, os.path.join(args.dump_dir, 'checkpoints'), "epoch_%d.pth" % (epoch + 1))
        else:
            current_reset_times = (epoch - args.max_epoches) // args.listener_reset_cycle + 1
            inner_epoch = (epoch - args.max_epoches) % args.listener_reset_cycle
            if inner_epoch == 0:
                logger.info("No.%d reset listener.\n" % current_reset_times)
                def weight_reset(m):
                    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                        m.reset_parameters()
                if torch.cuda.device_count() > 1:
                    model.module.listener.apply(weight_reset)
                else:
                    model.listener.apply(weight_reset)
                
                best_validation_acc = 0

             
            mean_train_acc, mean_train_loss = train(train_loader, optimizer, model, epoch)
            mean_validate_acc, mean_validate_loss = validate(validation_loader, model, epoch)
            mean_test_acc, mean_test_loss = test(test_loader, model, epoch)
            logger.info(
                """
                Epoch %d, generation %d:
                    mean train acc %.4f, mean train loss: %.4f
                    mean valid acc %.4f, mean valid loss: %.4f
                    mean test acc %.4f, mean test loss: %.4f
                """ % (
                    epoch, current_reset_times,
                    mean_train_acc, mean_train_loss, 
                    mean_validate_acc, mean_validate_loss,
                    mean_test_acc, mean_test_loss
                ))

            save_checkpoint(epoch, best_validation_acc, model, optimizer, os.path.join(args.dump_dir, 'checkpoints'), "last_epoch.pth")
            if mean_validate_acc > best_validation_acc:
                best_validation_acc = mean_validate_acc
                logger.info('Best validation acc: %.4f\n' % best_validation_acc)
                save_checkpoint(epoch, best_validation_acc, model, optimizer, os.path.join(args.dump_dir, 'checkpoints'), "best_epoch_gen%d.pth" % current_reset_times)
            if (epoch + 1) % args.listener_reset_cycle == 0:
                save_checkpoint(epoch, best_validation_acc, model, optimizer, os.path.join(args.dump_dir, 'checkpoints'), "epoch_%d.pth" % (epoch + 1))
        

def get_dataloader():
    train_set = RAVENDataset('train', args)
    test_set = RAVENDataset('test', args)
    validation_set = RAVENDataset('validation', args)
    train_loader = DataLoader(train_set, batch_size=args.train_batch_size, shuffle=True, num_workers=args.dataloader_num_workers)
    test_loader = DataLoader(test_set, batch_size=args.test_batch_size, shuffle=False, num_workers=args.dataloader_num_workers)
    validation_loader = DataLoader(validation_set, batch_size=args.validation_batch_size, shuffle=False, num_workers=args.dataloader_num_workers)

    return train_loader, test_loader, validation_loader

def get_parameter_groups(model):
    param_groups = [
        dict(params=model.speaker.parameters()),
        dict(params=model.listener.parameters()),
    ]
    return param_groups

def speaker_lr_schedule_lambda(epoch):
    base_lr = args.lr
    # if epoch < 7:
    #     return base_lr 
    # else:
    #     return base_lr 
    return 1e-10


def listener_lr_schedule_lambda(epoch):
    base_lr = args.lr
    factor = 2 ** min(epoch // 10, 4)
    # if epoch < 7:
    #     return base_lr
    # elif epoch < 20:
    #     return base_lr * 5
    # else:
    
    return 1e-10

def get_model():
    model = Model(args)
    return model

def get_cuda_model(model):
    model = model.cuda()
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    return model

def get_logger():
    # logging settings
    fh = logging.FileHandler(os.path.join(args.dump_dir, 'log.log'))
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('[%(asctime)s] ## %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger = logging.getLogger(__file__)
    logger.addHandler(fh)
    logger.addHandler(ch)
    logger.setLevel(logging.DEBUG)
    return logger

def train(dataloader, optimizer, model, epoch):
    num_iterations = len(dataloader)
    model.train()

    # # change Speaker lr
    # optimizer.param_groups[0]['lr'] = speaker_lr_schedule_lambda(epoch)
    # optimizer.param_groups[1]['lr'] = listener_lr_schedule_lambda(epoch)


    acc_total = 0
    loss_total = 0
    with tqdm(total=num_iterations) as pbar:
        for batch, input_dict in enumerate(dataloader):
            if args.visual:
                input_dict['image'] = input_dict['image'].cuda()
                input_dict['target_image'] = input_dict['target_image'].cuda()
            elif args.symbol:
                input_dict['symbol'] = input_dict['symbol'].cuda()
                input_dict['target_symbol'] = input_dict['target_symbol'].cuda()
            if args.rule:
                input_dict['rules'] = input_dict['rules'].cuda()
            input_dict['label'] = input_dict['label'].cuda()
            
            output_dict = model(input_dict)
            loss = output_dict['loss']
            loss.backward()
            # insepct grad
            # print(model)
            # print(model.speaker.agent.message_embedding.grad.mean().item(), model.speaker.agent.message_embedding.grad.max().item())
            # print(model.listener.agent.x.grad.mean().item(), model.listener.agent.x.grad.max().item())
            # print(model.speaker.agent.visual_model.cnn.conv_blocks[0].convs[0].weight.grad.mean().item(), model.speaker.agent.visual_model.cnn.conv_blocks[0].convs[0].weight.grad.max().item())
            # print(model.listener.agent.visual_model.cnn.conv_blocks[0].convs[0].weight.grad.mean().item(), model.listener.agent.visual_model.cnn.conv_blocks[0].convs[0].weight.grad.max().item())


            optimizer.step()
            optimizer.zero_grad()

            ## pred

            pred = output_dict['pred']
            batch_acc = output_dict['acc'].mean().item()
            batch_loss = loss.item()

            acc_total += batch_acc
            loss_total += batch_loss

            pbar.set_description('Train epoch: %d, acc: %.4f, loss: %.4f, main loss: %.4f' % (epoch, acc_total / (batch + 1), loss_total / (batch + 1), loss))
            pbar.update()
    return acc_total / num_iterations, loss_total / num_iterations

def test(dataloader, model, epoch):
    num_iterations = len(dataloader)
    model.eval()
    acc_total = 0
    loss_total = 0
    with torch.no_grad():
        with tqdm(total=num_iterations) as pbar:
            for batch, input_dict in enumerate(dataloader):
                if args.visual:
                    input_dict['image'] = input_dict['image'].cuda()
                    input_dict['target_image'] = input_dict['target_image'].cuda()
                elif args.symbol:
                    input_dict['symbol'] = input_dict['symbol'].cuda()
                    input_dict['target_symbol'] = input_dict['target_symbol'].cuda()
                if args.rule:
                    input_dict['rules'] = input_dict['rules'].cuda()

                input_dict['label'] = input_dict['label'].cuda()

                output_dict = model(input_dict)
                loss = output_dict['loss']

                ## pred
                pred = output_dict['pred']
                batch_acc = output_dict['acc'].mean().item()
                batch_loss = loss.item()

                acc_total += batch_acc
                loss_total += batch_loss
                pbar.set_description('Test epoch: %d, acc: %.4f, loss: %.4f' % (epoch, acc_total / (batch + 1), loss_total / (batch + 1)))
                pbar.update()
    return acc_total / num_iterations, loss_total / num_iterations


def validate(dataloader, model, epoch):
    num_iterations = len(dataloader)
    model.eval()
    acc_total = 0
    loss_total = 0
    with torch.no_grad():
        with tqdm(total=num_iterations) as pbar:
            for batch, input_dict in enumerate(dataloader):
                if args.visual:
                    input_dict['image'] = input_dict['image'].cuda()
                    input_dict['target_image'] = input_dict['target_image'].cuda()
                elif args.symbol:
                    input_dict['symbol'] = input_dict['symbol'].cuda()
                    input_dict['target_symbol'] = input_dict['target_symbol'].cuda()
                if args.rule:
                    input_dict['rules'] = input_dict['rules'].cuda()
                
                input_dict['label'] = input_dict['label'].cuda()
                output_dict = model(input_dict)
                loss = output_dict['loss']

                ## pred
                pred = output_dict['pred']
                batch_acc = output_dict['acc'].mean().item()
                batch_loss = loss.item()
                acc_total += batch_acc
                loss_total += batch_loss
                pbar.set_description('Valid epoch: %d, acc: %.4f, loss: %.4f' % (epoch, acc_total / (batch + 1), loss_total / (batch + 1)))
                pbar.update()
    return acc_total / num_iterations, loss_total / num_iterations


if __name__ == "__main__":
    
    if args.dump_message:
        if not args.listener_reset:
            cpkt = 'best_epoch'
            dump_message(cpkt=cpkt, message_dump_dir=0)
        else:
            gens = args.listener_reset_times
            for i in range(1, gens + 1):
                cpkt = 'best_epoch_gen%d' % i
                dump_message(cpkt=cpkt)
    else:
        main()