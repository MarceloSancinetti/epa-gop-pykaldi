import os
import glob
from pathlib import Path
import argparse
import yaml


import torchaudio
import torch
import torch.optim as optim
import torch.multiprocessing as mp

from finetuning_utils import *
from utils import *
from dataset import *

from torch.utils.data import DataLoader, ConcatDataset

from pytorch_models import *

import wandb

from IPython import embed

from sklearn.model_selection import KFold


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
    return torch.cuda.FloatTensor(phone_weights, device=device)

def get_path_for_checkpoint(state_dict_dir, run_name, fold, epoch):
    return state_dict_dir + run_name + '-fold-' + str(fold) + '-epoch-' + str(epoch) + '.pth'

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

#Handless all logs to wandb every i batches during training 
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


def start_from_checkpoint(PATH, model, optimizer):
    checkpoint = torch.load(PATH)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    step = checkpoint['step']
    return model, optimizer, step        

def freeze_layers_for_finetuning(model, layer_amount, use_dropout, use_first_bn):
    #Generate layer names for layers that should be trained
    layers_to_train = ['layer' + str(19 - x) for x in range(layer_amount)]

    #Freeze all layers except #layer_amount layers starting from the last
    for name, module in model.named_modules():
        freeze_layer = all([layer not in name for layer in layers_to_train])
        if freeze_layer:
            module.eval()
        else:
            module.train()

    if use_first_bn:
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


def calculate_loss(outputs, mask, labels, phone_weights=None, norm_per_phone=False):

    weights = mask *1
    
    if phone_weights is not None:
        weights = weights * phone_weights

    if norm_per_phone:
        weights = weights * torch.nan_to_num(1 / torch.sum(mask, dim=[0,1]))

    loss_fn = torch.nn.BCEWithLogitsLoss(reduction='none', weight=weights)

    return loss_fn(outputs, labels), torch.sum(weights)

def criterion_fast(batch_outputs, batch_labels, phone_weights=None, norm_per_phone=False, log_class_loss=False, phone_int2sym=None):

    batch_labels_for_loss = torch.abs((batch_labels-1)/2)

    loss_pos, weights_pos = calculate_loss(batch_outputs, batch_labels ==  1, batch_labels_for_loss, phone_weights=phone_weights, norm_per_phone=norm_per_phone)
    loss_neg, weights_neg = calculate_loss(batch_outputs, batch_labels == -1, batch_labels_for_loss, phone_weights=phone_weights, norm_per_phone=norm_per_phone)

    total_weights = weights_pos + weights_neg
    total_loss = (loss_pos + loss_neg).sum()


    if not norm_per_phone:
        #frame_count = torch.sum(batch_labels != 0)
        total_loss /= total_weights

    if log_class_loss:

        pos_phone_loss = torch.sum(loss_pos,dim=[0,1])
        neg_phone_loss = torch.sum(loss_neg,dim=[0,1])
        loss_dict = {}
        for phone, phone_sym in phone_int2sym.items():
            loss_dict[phone_sym+'+'] = pos_phone_loss[phone]/phone_weights[phone]/weights_pos
            loss_dict[phone_sym+'-'] = neg_phone_loss[phone]/phone_weights[phone]/weights_neg
  
        return total_loss, loss_dict
        
    else:
        return total_loss

def criterion_simple(batch_outputs, batch_labels):
    '''
    Calculates loss
    '''
    #embed()
    loss_fn = torch.nn.BCEWithLogitsLoss()
    #embed()
    batch_outputs, batch_labels = get_outputs_and_labels_for_loss(batch_outputs, batch_labels)
    #Calculate loss
    loss = loss_fn(batch_outputs, batch_labels)
    #embed()
    return loss


def train(model, trainloader, testloader, fold, epochs, state_dict_dir, run_name, layer_amount, use_dropout, lr, use_clipping, use_first_bn):
    global phone_weights, phone_count, device

    print("Started training fold " + str(fold))

    step = 0

    freeze_layers_for_finetuning(model, layer_amount, use_dropout, use_first_bn)

    optimizer = optim.Adam(model.parameters(), lr=lr)#, weight_decay=1e-5)

    for epoch in range(epochs):  # loop over the dataset multiple times
        PATH = get_path_for_checkpoint(state_dict_dir, run_name, fold, epoch) 
        #If the checkpoint for the current epoch is already present, checkpoint is loaded and training is skipped
        if os.path.isfile(PATH):
            model, optimizer, step = start_from_checkpoint(PATH, model, optimizer)
            continue

        running_loss = 0.0
        #loss_dict = {}
        for i, data in enumerate(trainloader, 0):            
            #print("Batch " + str(i))
            logids = unpack_logids_from_batch(data)
            inputs = unpack_features_from_batch(data).to(device)
            batch_labels = unpack_labels_from_batch(data).to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            outputs = model(inputs)
            #embed()

            loss = criterion_fast(outputs, batch_labels, phone_weights=phone_weights, phone_int2sym=phone_int2sym)
            #loss = criterion_simple(outputs, batch_labels)

            if epoch == 0 and i == 0:
                wandb.log({'train_loss_fold_' + str(fold): loss,
                          'step' : step})
                test_loss, test_loss_dict = test(model, testloader)
                step = log_test_loss(fold, test_loss, step, test_loss_dict)

            loss.backward()
            if use_clipping:
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0, error_if_nonfinite=True, norm_type=2)
            optimizer.step()

            #print statistics
            running_loss += loss.item()

            running_loss, step = log_and_reset_every_n_batches(fold, epoch, i, running_loss, step, 10)

        test_loss, test_loss_dict = test(model, testloader)
        step = log_test_loss(fold, test_loss, step, test_loss_dict)

        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'step': step
            }, PATH)


    return model

def test(model, testloader):

    global phone_weights, phone_count, phone_int2sym, device

    dataiter = iter(testloader)
    batch = dataiter.next()
    features = unpack_features_from_batch(batch).to(device)
    #pos_labels = unpack_pos_labels_from_batch(batch)
    #neg_labels = unpack_neg_labels_from_batch(batch)
    labels   = unpack_labels_from_batch(batch).to(device)

    outputs = model(features)
    loss_dict = {}
    loss, loss_dict = criterion_fast(outputs, labels, phone_weights=phone_weights, log_class_loss=True, phone_int2sym=phone_int2sym)
    #loss = criterion_simple(outputs, labels)

    loss = loss.item()

    return loss, loss_dict

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
    parser.add_argument('--utterance-list', dest='utterance_list', help='File with utt list', default=None)
    parser.add_argument('--folds', dest='fold_amount', help='Amount of folds to use in training', default=None)
    parser.add_argument('--epochs', dest='epoch_amount', help='Amount of epochs to use in training', default=None)
    parser.add_argument('--layers', dest='layer_amount', help='Amount of layers to train starting from the last (if layers=1 train only the last layer)', default=None)
    parser.add_argument('--learning-rate', dest='learning_rate', help='Learning rate to use during training', type=float, default=None)
    parser.add_argument('--batch-size', dest='batch_size', help='Batch size for training', type=int, default=None)
    parser.add_argument('--use-clipping', dest='use_clipping', help='Whether to use gradien clipping or not', default=None)
    parser.add_argument('--use-dropout', dest='use_dropout', help='Whether to unfreeze dropout components or not', default=None)
    parser.add_argument('--dropout-p', dest='dropout_p', help='Dropout probability', type=float, default=None)
    parser.add_argument('--use-first-batchnorm', dest='use_first_bn', help='Whether to use batch normalization on first layer or not', default=None)
    parser.add_argument('--use-final-batchnorm', dest='use_final_bn', help='Whether to use batch normalization on last layer or not', default=None)
    parser.add_argument('--phones-file', dest='phones_file', help='File with list of phones', default=None)
    parser.add_argument('--labels-dir', dest='labels_dir', help='Directory with labels used in training', default=None)
    parser.add_argument('--model-path', dest='model_path', help='Path to .pth/pt file with model to finetune', default=None)
    parser.add_argument('--phone-weights-path', dest='phone_weights_path', help='Path to .yaml containing weights for phone-level loss', default=None)
    parser.add_argument('--epa-root-path', dest='epa_root_path', help='EpaDB root path', default=None)
    parser.add_argument('--features-path', dest='features_path', help='Path to features directory', default=None)
    parser.add_argument('--conf-path', dest='conf_path', help='Path to config directory used in feature extraction', default=None)
    parser.add_argument('--test-sample-list-dir', dest='test_sample_list_dir', help='Path to output directory to save test sample lists', default=None)
    parser.add_argument('--state-dict-dir', dest='state_dict_dir', help='Path to output directory to save state dicts', default=None)
    parser.add_argument('--use-multi-process', dest='use_multi_process', help='Whether to use multiple processes or not', default=None)
    parser.add_argument('--device', dest='device_name', help='Device name to use, such as cpu or cuda', default=None)

    args              = parser.parse_args()
    run_name          = args.run_name
    device_name       = args.device_name
    folds             = int(args.fold_amount)
    epochs            = int(args.epoch_amount)
    layer_amount      = int(args.layer_amount)
    use_dropout       = parse_bool_arg(args.use_dropout)
    use_clipping      = parse_bool_arg(args.use_clipping)
    use_final_bn      = parse_bool_arg(args.use_final_bn)
    use_first_bn      = parse_bool_arg(args.use_first_bn)
    use_multi_process = parse_bool_arg(args.use_multi_process)

    wandb.init(project="gop-finetuning", entity="pronscoring-liaa")
    wandb.run.name = run_name

    epa_root_path = args.epa_root_path
    dataset = EpaDB(epa_root_path, args.utterance_list, args.phones_file, args.labels_dir, args.features_path, args.conf_path)

    global phone_int2sym, phone_weights, phone_count, device
    phone_int2sym = dataset.phone_int2sym_dict

    device = torch.device(device_name)

    seed = 42
    torch.manual_seed(seed)

    kfold = KFold(n_splits=folds, shuffle=True, random_state = seed)

    spkr_list = dataset.get_speaker_list()

    phone_weights = get_phone_weights_as_torch(args.phone_weights_path)

    for fold, (train_spkr_indexes, test_spkr_indexes) in enumerate(kfold.split(spkr_list)):


        train_sample_indexes = dataset.get_sample_indexes_from_spkr_indexes(train_spkr_indexes)
        test_sample_indexes  = dataset.get_sample_indexes_from_spkr_indexes(test_spkr_indexes)

        train_subsampler = torch.utils.data.SubsetRandomSampler(train_sample_indexes)
        test_subsampler  = torch.utils.data.SubsetRandomSampler(test_sample_indexes)

        trainloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size,
                                     num_workers=1, sampler=train_subsampler, collate_fn=collate_fn_padd)

        testloader = torch.utils.data.DataLoader(dataset, batch_size=32, 
                                     num_workers=1, sampler=test_subsampler, collate_fn=collate_fn_padd)

        phone_count = dataset.phone_count()

        #Get acoustic model to train
        model = FTDNN(out_dim=phone_count, use_final_bn=use_final_bn, use_first_bn=use_first_bn, dropout_p=args.dropout_p, device_name=device_name) 
        model.to(device)
        state_dict = torch.load(get_model_path_for_fold(args.model_path, fold, layer_amount))
        model.load_state_dict(state_dict['model_state_dict'])

        #Train the model
        wandb.watch(model, log_freq=100)
        if use_multi_process:
            processes = []
            p = mp.Process(target=train, args=(model, trainloader, testloader, fold, 
                           epochs, args.state_dict_dir, run_name, layer_amount, use_dropout, 
                           args.learning_rate, use_clipping))
            p.start()
            processes.append(p)
        else:
            train(model, trainloader, testloader, fold, epochs, args.state_dict_dir,
                  run_name, layer_amount, use_dropout, args.learning_rate, use_clipping, use_first_bn)

        #Generate test sample list for current fold
        generate_test_sample_list(testloader, epa_root_path, args.test_sample_list_dir, 'test_sample_list_fold_' + str(fold))

    if use_multi_process:
        for p in processes:
            p.join()

if __name__ == '__main__':
    main()
