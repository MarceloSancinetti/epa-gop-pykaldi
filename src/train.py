import os
import glob
from pathlib import Path
import argparse
import yaml
import time

import torchaudio
import torch
import torch.optim as optim
from torch.optim.swa_utils import AveragedModel, SWALR
import torch.multiprocessing as mp

from finetuning_utils import *
from utils import *
from dataset import *

from torch.utils.data import DataLoader, ConcatDataset

from pytorch_models import *

import wandb

from IPython import embed

global phone_int2sym

def get_model_path_for_fold(model_path, fold, layer_amount):
    #This is used to allow training to start from a previous experiment's
    #state_dict with the same fold
    return model_path.replace("@FOLD@", str(fold)) 

def get_phone_weights_as_torch(phone_weights_path):
    global device
    phone_weights_fh = open(phone_weights_path)
    phone_weights = yaml.safe_load(phone_weights_fh)
    weights_list = []
    for phone, weight in phone_weights.items():
        weights_list.append(weight)
    phone_weights = weights_list
    return torch.tensor(phone_weights, device=device)

def get_path_for_checkpoint(state_dict_dir, run_name, fold, epoch, suffix=''):
    if "heldout" in run_name:
        fold_identifier = ''
    else:
        fold_identifier = '-fold-' + str(fold)
    return state_dict_dir + run_name + fold_identifier + '-epoch-' + str(epoch) + suffix + '.pth'

#Logs loss for each phone in the loss dict to wandb 
def get_log_dict_for_wandb_from_loss_dict(fold, loss_dict, tag):
    log_dict = {}
    for phone, loss in loss_dict.items():
        key = tag + '_loss_fold_' + str(fold) + '_phone_' + phone 
        log_dict[key] = loss
    return log_dict


#Logs loss on test set 
def log_test_loss(fold, loss, step, loss_dict):
    log_dict = {'test_loss_fold_' + str(fold): loss,
               'step' : step}
    loss_dict = {phone : loss for phone, loss in loss_dict.items()}
    log_dict.update(get_log_dict_for_wandb_from_loss_dict(fold, loss_dict, 'test'))
    wandb.log(log_dict)
    step += 1
    return step

#Handles all logs to wandb every i batches during training 
def log_and_reset_every_n_batches(fold, epoch, i, running_loss, step, n):
    if i % n == n-1:    # log every i mini-batches
        print('Fold ' + str(fold), ' Epoch ' + str(epoch) + ' Batch ' + str(i))
        print('running_loss ' + str(running_loss/n))
        log_dict = {'train_loss_fold_' + str(fold): running_loss/n,
                   'step' : step,
                   'epoch': epoch}
        #loss_dict = {phone : loss/n for phone, loss in loss_dict.items()}
        #log_dict.update(get_log_dict_for_wandb_from_loss_dict(fold, loss_dict, 'train'))
        wandb.log(log_dict)
        step += 1
        running_loss = 0.0
    return running_loss, step#, loss_dict      

def freeze_layers_for_finetuning(model, layer_amount, use_dropout, batchnorm):
    #Generate layer names for layers that should be trained
    layers_to_train = ['layer' + str(19 - x) for x in range(layer_amount)]

    #Freeze all layers except #layer_amount layers starting from the last
    for name, module in model.named_modules():
        freeze_layer = all([layer not in name for layer in layers_to_train])
        if freeze_layer and (batchnorm != 'all' or 'bn' not in name):
            module.eval()
        else:
            module.train()

    if batchnorm == 'first' or batchnorm=='firstlast':
        model.layer01.bn.train()

    for name, param in model.named_parameters():
        freeze_layer = all([layer not in name for layer in layers_to_train])
        if freeze_layer:
            param.requires_grad = False

    #Unfreeze dropouts
    for name, module in model.named_modules():
        if 'dropout' in name and use_dropout:
            module.train()

def freeze_layers_for_finetuning_buggy(model, layer_amount):
    #Generate layer names for layers that should be trained
    layers_to_train = ['layer' + str(19 - x) for x in range(layer_amount)]

    #Freeze all layers except #layer_amount layers starting from the last
    for name, param in model.named_parameters():
        freeze_layer = all([layer not in name for layer in layers_to_train])
        if freeze_layer:
            param.requires_grad = False

#Builds a dictionary with the individual loss for each phone/class for logging purposes
def add_loss_for_phone_to_dict(loss, phone, loss_dict, label):
    key = phone + label
    if key in loss_dict.keys():
        loss_dict[key] = loss_dict[key] + loss
    else:
        loss_dict[key] = loss
        
    return loss_dict
        

#This function calculates the loss for a specific phone in the phone set given the outputs and labels
def loss_for_phone(outputs, labels, phone, stop=False):
    global phone_weights, phone_count

    loss_fn = torch.nn.BCEWithLogitsLoss(reduction='mean')
    phone_mask = torch.zeros(phone_count)
    phone_mask[phone] = 1
    labels_for_phone = labels * phone_mask

    outputs, labels = get_outputs_and_labels_for_loss(outputs, labels_for_phone)
    #Calculate loss
    occurrences = labels.shape[0]
    phone_weight = phone_weights[phone]
    if occurrences == 0:
        return 0
    else:
        return loss_fn(outputs, labels) * phone_weight

#Returns total batch loss, adding the loss computed for each class individually
def criterion_slow(batch_outputs, batch_pos_labels, batch_neg_labels, loss_dict):
    '''
    Calculates loss
    '''
    global phone_int2sym, phone_weights, phone_count

    total_loss = 0
    for phone_int in range(phone_count):
        phone_sym = phone_int2sym[phone_int]        
        
        phone_loss  = loss_for_phone(batch_outputs, batch_pos_labels, phone_int) 
        loss_dict   = add_loss_for_phone_to_dict(phone_loss, phone_sym, loss_dict, '+')
        total_loss += phone_loss
        #embed()

        phone_loss  = loss_for_phone(batch_outputs, batch_neg_labels, phone_int, stop=True)
        loss_dict   = add_loss_for_phone_to_dict(phone_loss, phone_sym, loss_dict, '-')    
        total_loss += phone_loss    
        #embed()    
    
    return total_loss, loss_dict


def calculate_loss(outputs, mask, labels, phone_weights=None, norm_per_phone_and_class=False, min_frame_count=0):

    weights = mask *1

    if phone_weights is not None:
        weights = weights * phone_weights

    if norm_per_phone_and_class:
        frame_count = torch.sum(mask, dim=[0,1])
        weights = weights * torch.nan_to_num(1 / frame_count)
        if min_frame_count > 0:
            # Set to 0 the weights for the phones with too few cases in this batch
            weights[:,:,frame_count<min_frame_count] = 0.0

    loss_fn = torch.nn.BCEWithLogitsLoss(reduction='none', weight=weights)

    return loss_fn(outputs, labels), torch.sum(weights)


def criterion_fast(batch_outputs, batch_labels, weights=None, norm_per_phone_and_class=False, log_per_phone_and_class_loss=False, phone_int2sym=None, min_frame_count=0):
    
    batch_labels_for_loss = torch.abs((batch_labels-1)/2)

    loss_pos, sum_weights_pos = calculate_loss(batch_outputs, batch_labels ==  1, batch_labels_for_loss, phone_weights=weights,  norm_per_phone_and_class=norm_per_phone_and_class, min_frame_count=min_frame_count)
    loss_neg, sum_weights_neg = calculate_loss(batch_outputs, batch_labels == -1, batch_labels_for_loss, phone_weights=weights, norm_per_phone_and_class=norm_per_phone_and_class, min_frame_count=min_frame_count)

    total_loss = (loss_pos + loss_neg).sum()

    if not norm_per_phone_and_class:
        total_weights = sum_weights_pos + sum_weights_neg
        total_loss /= total_weights

    if log_per_phone_and_class_loss:

        pos_phone_loss = torch.sum(loss_pos,dim=[0,1])
        neg_phone_loss = torch.sum(loss_neg,dim=[0,1])
        loss_dict = {}
        for phone, phone_sym in phone_int2sym.items():
            loss_dict[phone_sym+'+'] = pos_phone_loss[phone]/weights[phone]/sum_weights_pos
            loss_dict[phone_sym+'-'] = neg_phone_loss[phone]/weights[phone]/sum_weights_neg

        return total_loss, loss_dict

    else:
        return total_loss

def criterion_simple(batch_outputs, batch_labels):
    '''
    Calculates loss
    '''
    loss_fn = torch.nn.BCEWithLogitsLoss()
    batch_outputs, batch_labels = get_outputs_and_labels_for_loss(batch_outputs, batch_labels)
    loss = loss_fn(batch_outputs, batch_labels)
    return loss

def start_from_checkpoint(PATH, model, optimizer):
    checkpoint = torch.load(PATH)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    step = checkpoint['step']
    return model, optimizer, step

def save_state_dict(state_dict_dir, run_name, fold, epoch, step, model, optimizer, suffix=''):
    PATH = get_path_for_checkpoint(state_dict_dir, run_name, fold, epoch, suffix=suffix) 
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'step': step
    }, PATH)

def choose_starting_epoch(epochs, state_dict_dir, run_name, fold, model, optimizer):
    step = 0 #wandb step
    # Look in the output dir for any existing check points, from last to first.
    start_from_epoch = 0
    for epoch in range(epochs, 0, -1):
        PATH = get_path_for_checkpoint(state_dict_dir, run_name, fold, epoch) 
        if os.path.isfile(PATH):
            model, optimizer, step = start_from_checkpoint(PATH, model, optimizer)
            start_from_epoch = epoch+1
            print("Loaded pre-existing checkpoint for epoch %d (%s)"% (epoch, PATH))            
            break

    if start_from_epoch == 0:
        # Save the initial model
        save_state_dict(state_dict_dir, run_name, fold, 0, step, model, optimizer)


    return model, optimizer, step, start_from_epoch

def foward_backward_pass(data, model, optimizer, phone_weights, phone_int2sym, norm_per_phone_and_class):
    logids       = unpack_logids_from_batch(data)
    inputs       = unpack_features_from_batch(data).to(device)
    batch_labels = unpack_labels_from_batch(data).to(device)
    
    # zero the parameter gradients
    optimizer.zero_grad()
    
    outputs = model(inputs)
    
    loss = criterion_fast(outputs, batch_labels, weights=phone_weights, phone_int2sym=phone_int2sym, norm_per_phone_and_class=norm_per_phone_and_class, min_frame_count=0)

    loss.backward()

    return loss

def log_loss_if_first_batch(epoch, i, fold, loss, model, testloader, step):
    if epoch == 0 and i == 0:
        wandb.log({'train_loss_fold_' + str(fold): loss,
                  'step' : step})
        test_loss, test_loss_dict = test(model, testloader)
        step = log_test_loss(fold, test_loss, step, test_loss_dict)

    return step

def train_one_epoch(trainloader, testloader, model, optimizer, running_loss, fold, epoch, step, use_clipping, phone_weights, phone_int2sym, norm_per_phone_and_class):

    for i, data in enumerate(trainloader, 0):

        loss = foward_backward_pass(data, model, optimizer, phone_weights, phone_int2sym, norm_per_phone_and_class)

        step = log_loss_if_first_batch(epoch, i, fold, loss, model, testloader, step)

        if use_clipping:
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0, error_if_nonfinite=True, norm_type=2)
        
        optimizer.step()

        running_loss += loss.item()

        running_loss, step = log_and_reset_every_n_batches(fold, epoch, i, running_loss, step, 10)

    return running_loss, step, model, optimizer

def train(model, trainloader, testloader, fold, epochs, swa_epochs, state_dict_dir, run_name, layer_amount, use_dropout, lr, swa_lr, use_clipping, batchnorm, norm_per_phone_and_class):
    global phone_weights, phone_count, device

    print("Started training fold " + str(fold))

    freeze_layers_for_finetuning(model, layer_amount, use_dropout, batchnorm)

    optimizer = optim.Adam(model.parameters(), lr=lr)

    #Find the most advanced state dict to start from
    model, optimizer, step, start_from_epoch = choose_starting_epoch(epochs, state_dict_dir, run_name, 
                                                                     fold, model, optimizer)

    swa_model  = AveragedModel(model)
    swa_start  = epochs - swa_epochs
    if swa_epochs > 0:
        swa_scheduler = SWALR(optimizer, swa_lr=swa_lr)

    for epoch in range(start_from_epoch, epochs):  # loop over the dataset multiple times

        running_loss = 0.0
        
        running_loss, step, model, optimizer = train_one_epoch(trainloader, testloader, model, optimizer, running_loss, fold, epoch,
                                                        step, use_clipping, phone_weights, phone_int2sym, norm_per_phone_and_class)

        test_loss, test_loss_dict = test(model, testloader)
        step = log_test_loss(fold, test_loss, step, test_loss_dict)

        if epoch >= swa_start:
            swa_model.update_parameters(model)
            swa_scheduler.step()
            if epoch % 25 == 24:
                save_state_dict(state_dict_dir, run_name, fold, epoch+1, step, swa_model, optimizer, suffix='_swa')


        if epoch % 25 == 24:
            save_state_dict(state_dict_dir, run_name, fold, epoch+1, step, model, optimizer)


def test(model, testloader):

    global phone_weights, phone_count, phone_int2sym, device

    total_loss = 0
    for i, batch in enumerate(testloader, 0):
        features = unpack_features_from_batch(batch).to(device)
        #pos_labels = unpack_pos_labels_from_batch(batch)
        #neg_labels = unpack_neg_labels_from_batch(batch)
        labels   = unpack_labels_from_batch(batch).to(device)

        outputs = model(features)
        loss_dict = {}
        loss, loss_dict = criterion_fast(outputs, labels, weights=phone_weights, 
                                         log_per_phone_and_class_loss=True, phone_int2sym=phone_int2sym)
        #loss = criterion_simple(outputs, labels)

        loss = loss.item()
        total_loss += loss

    return total_loss / (i + 1), loss_dict

def parse_bool_arg(arg):
    if arg not in ["true", "false"]:
        raise Exception("Argument must be true or false, got " + arg)

    if arg == "true":
        arg = True
    else:
        arg = False

    return arg

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--run-name', dest='run_name', help='Run name', default=None)
    parser.add_argument('--trainset-list', dest='trainset_list', help='File with trainset utt list', default=None)
    parser.add_argument('--testset-list', dest='testset_list', help='File with testset utt list', default=None)
    parser.add_argument('--fold', dest='fold', help='Fold number', type=int, default=None)
    parser.add_argument('--epochs', dest='epoch_amount', help='Amount of epochs to use in training', type=int, default=None)
    parser.add_argument('--swa-epochs', dest='swa_epochs', help='Amount of SWA epochs to use in training', type=int, default=0)
    parser.add_argument('--layers', dest='layer_amount', help='Amount of layers to train starting from the last (if layers=1 train only the last layer)', type=int, default=None)
    parser.add_argument('--learning-rate', dest='learning_rate', help='Learning rate to use during training', type=float, default=None)
    parser.add_argument('--swa-learning-rate', dest='swa_lr', help='Learning rate to use during SWA training', type=float, default=None)
    parser.add_argument('--batch-size', dest='batch_size', help='Batch size for training', type=int, default=None)
    parser.add_argument('--norm-per-phone-and-class', dest='norm_per_phone_and_class', help='Whether to normalize phone level loss by frame count or not', default=None)
    parser.add_argument('--use-clipping', dest='use_clipping', help='Whether to use gradient clipping or not', default=None)
    parser.add_argument('--use-dropout', dest='use_dropout', help='Whether to unfreeze dropout components or not', default=None)
    parser.add_argument('--dropout-p', dest='dropout_p', help='Dropout probability', type=float, default=None)
    parser.add_argument('--batchnorm', dest='batchnorm', help='Batchnorm option (first, final, all)', default=None)
    parser.add_argument('--phones-file', dest='phones_file', help='File with list of phones', default=None)
    parser.add_argument('--labels-dir', dest='labels_dir', help='Directory with labels used in training', default=None)
    parser.add_argument('--model-path', dest='model_path', help='Path to .pth/pt file with model to finetune', default=None)
    parser.add_argument('--phone-weights-path', dest='phone_weights_path', help='Path to .yaml containing weights for phone-level loss', default=None)
    parser.add_argument('--train-root-path', dest='train_root_path', help='EpaDB root path', default=None)
    parser.add_argument('--test-root-path', dest='test_root_path', help='EpaDB root path', default=None)
    parser.add_argument('--features-path', dest='features_path', help='Path to features directory', default=None)
    parser.add_argument('--conf-path', dest='conf_path', help='Path to config directory used in feature extraction', default=None)
    parser.add_argument('--test-sample-list-dir', dest='test_sample_list_dir', help='Path to output directory to save test sample lists', default=None)
    parser.add_argument('--state-dict-dir', dest='state_dict_dir', help='Path to output directory to save state dicts', default=None)
    parser.add_argument('--use-multi-process', dest='use_multi_process', help='Whether to use multiple processes or not', default=None)
    parser.add_argument('--device', dest='device_name', help='Device name to use, such as cpu or cuda', default=None)

    args                     = parser.parse_args()
    run_name                 = args.run_name
    device_name              = args.device_name
    layer_amount             = args.layer_amount
    epochs                   = args.epoch_amount
    swa_epochs               = args.swa_epochs
    fold                     = args.fold
    use_dropout              = parse_bool_arg(args.use_dropout)
    use_clipping             = parse_bool_arg(args.use_clipping)
    use_multi_process        = parse_bool_arg(args.use_multi_process)
    norm_per_phone_and_class = parse_bool_arg(args.norm_per_phone_and_class)

    wandb.init(project="gop-finetuning", entity="pronscoring-liaa")
    wandb.run.name = run_name

    train_root_path = args.train_root_path
    test_root_path = args.test_root_path

    trainset = EpaDB(train_root_path, args.trainset_list, args.phones_file, args.labels_dir, args.features_path, args.conf_path)
    testset  = EpaDB(test_root_path,  args.testset_list , args.phones_file, args.labels_dir, args.features_path, args.conf_path)

    global phone_int2sym, phone_weights, phone_count, device
    phone_int2sym = trainset.phone_int2sym_dict

    device = torch.device(device_name)

    seed = 42
    torch.manual_seed(seed)

    phone_weights = get_phone_weights_as_torch(args.phone_weights_path)


    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                 num_workers=0, collate_fn=collate_fn_padd, shuffle=True)

    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, 
                                 num_workers=0, collate_fn=collate_fn_padd)

    phone_count = trainset.phone_count()

    #Get acoustic model to train
    model = FTDNN(out_dim=phone_count, batchnorm=args.batchnorm, dropout_p=args.dropout_p, device_name=device_name) 
    model.to(device)
    state_dict = torch.load(get_model_path_for_fold(args.model_path, fold, layer_amount))
    model.load_state_dict(state_dict['model_state_dict'])

    #Train the model
    wandb.watch(model, log_freq=100)
    train(model, trainloader, testloader, fold, epochs, swa_epochs, args.state_dict_dir, run_name, layer_amount, 
          use_dropout, args.learning_rate, args.swa_lr, use_clipping, args.batchnorm, norm_per_phone_and_class)

if __name__ == '__main__':
    main()
