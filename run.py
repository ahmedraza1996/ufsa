#!/usr/bin/python2.7
# code from https://github.com/yabufarha/ms-tcn (MIT+CC License) - many modifications
import argparse
import datetime
import json
import os
import random
import subprocess
import sys

import numpy as np
import pytz
import torch
import wandb
from dataloder import DataGenerator as DataGenerator
from model import Trainer
import logging



logger = logging.getLogger(__name__)


def initiate_wandb(args):
    wandb.init(project=args.project, entity=args.entity, name=args.wandb_name)
    wandb.config.update(vars(args))



parser = argparse.ArgumentParser()
#wandb arguements
parser.add_argument('--project' , default = 'uvast_uas' , type = str, help ='wand project name')
parser.add_argument('--entity' , default = 'ahmedrazamehmood' , type = str, help ='wand username')
parser.add_argument('--log_wandb', default=False, action='store_true', help='log wandb data')

# general arguments
parser.add_argument('--features_dim', default='64', type=int, help='dimension of input features (I3D features are 2048 dim)')
parser.add_argument('--experiment_path', default='logs/',  type=str, help='directory to save experiment results')
# dataset and dataloader
parser.add_argument('--split', default='1', type=int, help='split of the dataset')
parser.add_argument('--parent_dir' , default = '' , type = str, help ='root to project folder')
parser.add_argument('--data_root', default='/home/ahmed/Ahmed_data/UVAST/UVAST/data',  type=str, help='root to the datasets directory')
parser.add_argument('--data_root_mean_duration', default='/path_to_data_directory/',  type=str, help='root to the mean durations of the datasets')
parser.add_argument('--features_path', default=None, type=str)
parser.add_argument('--feature_type', default=".npy", type=str)
parser.add_argument('--feature_mode', default='feature', type=str)
parser.add_argument('--num_classes', default=-1, type=int, help='number of classes for each dataset (gtea=11, salads=19, breakfast=48)')
parser.add_argument('--skip_inference', default=False, action='store_true', help='skip inference')
parser.add_argument('--do_timing', default=False, action='store_true', help='do time measuring')
parser.add_argument('--dataset', default='50salads', type=str)
parser.add_argument('--sample_rate', default=1, type=int, help='frame sampling rate (salad=2, beakfast,gtea=1)')
parser.add_argument('--aug_rnd_drop', default=False, action='store_true', help='optional augmention for training: randomly dropping frames')
parser.add_argument('--split_segments', default=False, action='store_true', help='splitting segments')
parser.add_argument('--split_segments_max_dur', default='0.1', type=float, help='max duration in split_segments; for details see the paper')
parser.add_argument('--use_cuda', action="store_true",help='use cuda for training')
parser.add_argument('--exp_name', default='fs_test', type=str,help='name of the experiment, specifies the folder where models are saved')
parser.add_argument('--seed',  default=None, type=int, help='specify seed number')
parser.add_argument('--save_args', default=True, action='store_true', help='save arguments that were used for training')
parser.add_argument('--bg', default=False, action='store_true', help='for YTI, if dataset has background')
# testing
parser.add_argument('--inference_only', default=False, action='store_true', help='run inference only')
parser.add_argument('--inference_epochs', default=5, type = int, help='run inference at every N epochs')
parser.add_argument('--checkpoint_epoch', default=5, type = int, help='save training model at every N epochs')

parser.add_argument('--path_inference_model', default=None, type=str, help='path to model for inference')
parser.add_argument('--epoch_num_for_testing', default=10, type=int,help='evaluate specific epoch')
parser.add_argument('--do_refine_actions_no_consecutive_similar_action', action='store_true', default=False, help='remove duplicate actions during viterbi inference')
# encoder arguments
parser.add_argument('--num_f_maps', default='30', type=int, help='feature dimension in the transformer')
parser.add_argument('--activation', default='relu', type=str, help='type of activation used in the enc and dec')
parser.add_argument('--encoder_model', default='asformer_advanced', type=str, choices=['asformer_advanced', 'asformer_org_enc', 'asformer_org_enc_dec' ,"mlp"], help='select encoder model')
parser.add_argument('--enc_norm_type', default='InstanceNorm1d', type=str, help=['LayerNorm', 'InstanceNorm1d_track', 'InstanceNorm1d'])
parser.add_argument('--num_layers_enc',  default=2, type=int, help='the number of encoder layers in ASFormer')
parser.add_argument('--num_layers_asformer_dec',  default=2, type=int, help='number of decoder layers in the ASFormer (only used with asformer_org_enc_dec)')

# transcript decoder arguments
parser.add_argument('--use_transcript_dec', default=False, action='store_true', help='use transcript decoder for transcript prediction (second stage)')
parser.add_argument('--n_head_dec', default=1, type=int, help='the number of heads of the transcript decoder (first stage)')
parser.add_argument('--AttentionPoolType_dec', default='none', type=str,help='option to smooth the cross attention; options are: none, avg, max')
parser.add_argument('--AttentionPoolKernel_dec', default=31, type=int,help='kernel size for smoothing if --AttentionPoolType_dec is selected')
parser.add_argument('--num_layers_dec', default=2, type=int,help='number of layers of the transcript decoder (first stage)')
parser.add_argument('--dropout', default=0.1, type=float, help='dropout for the transcript decoder')
parser.add_argument('--use_pe_tgt', default=True, action='store_true', help='use positional encoding for target in the transcript decoder')
parser.add_argument('--use_pe_src', default=False, action='store_true', help='use positional encoding for source in the transcript decoder')
parser.add_argument('--dec_dim_feedforward', default=2048, type=int,help='similar to dec_dim_feedforward option in the standard pytorch model')
parser.add_argument('--len_seg_max',  default=7, type=int, help='the max number of action segments')
# optimizer arguments
parser.add_argument('--bs', default='1', type=int,help='batch size')
parser.add_argument('--usebatch', default=False, action='store_true', help='if bs > 1 is used the samples need to be padded')
parser.add_argument('--num_epochs', default=100, type=int, help='number of epochs')
parser.add_argument('--save_plot_epochs', default=1, type = int, help='save proto plots at every N epoch')
parser.add_argument('--save_plot', default=True, action='store_true', help='if we want to save proto plots')
parser.add_argument('--save_seg', default=False, action='store_true', help='if we want to save segmentation plots')
parser.add_argument('--step_size', default='400', type=int,help='the epoch when the lr will drop by factor of 10')
parser.add_argument('--lr_scheduler', default=True, action='store_true', help='use reduce on plateau lr scheduler')
parser.add_argument('--learning_rate', default=1e-3, type=float, help='learning rate')
parser.add_argument('--gamma', default=0.1, type=float, help='gamma for the optimizer')
parser.add_argument('--weight_decay', default=1e-5, type=float, help='weight decay')
parser.add_argument('--optimizer', default='adam', type=str, help='optimizer type')
parser.add_argument('--adap_clip_gradient', default=False, action='store_true', help='use gradient clipping')
parser.add_argument('--freeze_epochs', default=20, type=int, help='number of iteration to keep prototype layer of encoder freezed')
parser.add_argument('--epsilon', default=0.07, type=float, help='number of iteration to keep prototype layer of encoder freezed')
parser.add_argument('--sigma', default=0.75, type=float, help='sigma for p_gauss')
parser.add_argument('--num_videos', default=1, type=int, help='number of videos per batch')
parser.add_argument('--gaussian_temperature', default=0.1, type=float, help='temperature for proto scores')
parser.add_argument('--clip_percentile', default=10, type=int, help='adaptive gradient clipping is based on percentile of the previous gradient')
#TOT arguments
parser.add_argument("--apply_order_preserving_formulation", default=False, type=bool, help="Add Order Preserving Wasserstein Distance Prior")
parser.add_argument("--probability_threshold", default=0.35, type=float)
parser.add_argument("--background_frame", default=False, type=bool)
parser.add_argument("--N", default=3, type=int, help = "iterations of sinkhorn")

# loss arguments
parser.add_argument('--temperature', default=1e-3, type=float, help='the temperature in the cross attention loss')
parser.add_argument('--do_framewise_loss_g', default=False, action='store_true', help='use group wise frame wise loss')
parser.add_argument('--do_framewise_loss_gauss', default=False, action='store_true', help='use frame wise loss with gaussian')
parser.add_argument('--do_framewise_loss', default=False, action='store_true', help='use group wise frame wise loss')
parser.add_argument('--do_framewise_alignment_loss', default=False, action='store_true', help='use frame wise loss for alignment')
parser.add_argument('--do_segwise_loss', default=False, action='store_true', help='use segment wise ce loss')
parser.add_argument('--do_segwise_loss_g', default=False, action='store_true', help='use group segment wise ce loss')
parser.add_argument('--do_crossattention_action_loss_nll', default=False, action='store_true', help='use cross attention loss for first stage')
parser.add_argument('--do_crossattention_dur_loss_ce', default=False, action='store_true', help='use cross attention loss for second stage')
parser.add_argument('--framewise_loss_g_apply_logsoftmax', default=False, action='store_true', help='type of the normalization for group wise CE') 
parser.add_argument('--framewise_loss_g_apply_nothing', default=False, action='store_true', help='type of the normalization for group wise CE-this is normal averaging')
parser.add_argument('--segwise_loss_g_apply_logsoftmax', default=False, action='store_true', help='type of the normalization for group wise CE')  
parser.add_argument('--segwise_loss_g_apply_nothing', default=False, action='store_true', help='type of the normalization for group wise CE-this is normal averaging') 
parser.add_argument('--fixorder_T', default=False, action='store_true', help='if we want fix order instead of permutation aware in stage 2') 

args = parser.parse_args()
# #############################################################################################################################################################################################


device = torch.device("cuda" if torch.cuda.is_available() and args.use_cuda else "cpu")
device = torch.device('cuda')
if args.dataset == 'gtea':
    args.num_classes = 11
    args.sample_rate = 1
    args.channel_masking_rate = 0.5
    args.split_segments_max_dur = 0.17
elif args.dataset == '50salads':
    args.num_classes = 12
    args.sample_rate = 1
    args.channel_masking_rate = 0.3
    args.split_segments_max_dur = 0.15
    
elif args.dataset == 'breakfast':
    args.num_classes =5
    args.sample_rate = 1
    args.channel_masking_rate = 0.1
    args.split_segments_max_dur = 0.17
elif args.dataset == 'desktop_assembly' or args.dataset == 'desktop_assembly_v3':
    args.num_classes = 23
    args.sample_rate = 1
    args.channel_masking_rate = 0.3
    args.split_segments_max_dur = 0.17
elif args.dataset == 'yti' :
    args.num_classes = 11
    args.sample_rate = 1
    args.channel_masking_rate = 0.3
    args.split_segments_max_dur = 0.17
args.len_seg_max = args.num_classes+2
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

print(subprocess.check_output(["conda", "--version"]))
print('pytorch version {}', torch.__version__)
print('numpy version {}', np.__version__)
print('cuda', torch.version.cuda)
print('cudnn', torch.backends.cudnn.version())
print('device is:', device)


args.parent_dir = os.path.abspath(os.path.join(args.data_root, os.pardir))
args.pretrained_model ='/home/ahmed/Ahmed_data/Benchmark results/FS/fs_eval/fs_stage2_test_ca/model/50salads/epoch-20.model'
args.path_inference_model ='/home/ahmed/Ahmed_data/Benchmark results/FS/fs_eval/fs_stage2_test_ca/model/50salads/epoch-20.model'

#args.pretrained_model ="/home/ahmed/Ahmed_data/Benchmark results/DA_v1/da_stage1_test/model/desktop_assembly/epoch-400.model"
#args.pretrained_model= "/home/ahmed/Ahmed_data/UVAST/UVAST/figures/dav1_figure/model/desktop_assembly/epoch-1.model"
#args.path_inference_model= "/home/ahmed/Ahmed_data/UVAST/UVAST/figures/dav1_figure/model/desktop_assembly/epoch-1.model"
parent_path =''
if args.exp_name:
    #exp_path = parent_path +'/' + args.experiment_path +'/' + args.exp_name
   
    exp_path = os.path.join(parent_path,args.experiment_path, args.exp_name)
    args.results_path = parent_path  + args.experiment_path +  args.exp_name + '/'+'results'
    #args.results_path = os.path.join(parent_path,args.experiment_path, args.exp_name, 'results')
    print("results_paths : ", args.results_path)
else:
    print('Oops!! You did not provide exp_name and it can cause your experiments results be saved in one folder and cause issues later!')
    args.results_path = parent_path + '/'+ args.experiment_path+ '/results'
model_dir_base = args.results_path.replace('results', 'model')
model_dir = os.path.join(model_dir_base , args.dataset)
print(model_dir_base)
if not os.path.exists(os.path.join(parent_path, model_dir)):
        
        os.makedirs(os.path.join(model_dir), mode=0o777)
print("model path: " , model_dir)
results_dir = os.path.join(args.results_path, args.dataset)
args.results_dir = results_dir
args.wandb_name = args.exp_name + datetime.datetime.now(pytz.timezone('US/Eastern')).isoformat()
plots_dir = os.path.join(args.parent_dir, args.experiment_path,args.exp_name ,"plots")
if not os.path.exists(plots_dir): 
    os.mkdir(plots_dir)
if not os.path.exists(os.path.join(parent_path,results_dir)) and not args.inference_only:
    os.makedirs(os.path.join(parent_path,results_dir))

print('results_dir', results_dir)
print('model_dir', model_dir)
args.model_dir = model_dir
predictions_dir = os.path.join(exp_path,'predictions')
args.predictions_dir = predictions_dir
if not os.path.exists(predictions_dir): 
    os.mkdir(predictions_dir)
#print('start time:')
#print(datetime.datetime.now(pytz.timezone('US/Eastern')).isoformat())

# saving args to a file
if args.save_args and not args.inference_only:
    json_name = os.path.join(model_dir, 'args.txt')
    with open(json_name, 'w') as f:
        json.dump(args.__dict__, f, indent=2)

args.device = device
torch.set_default_dtype(torch.float64)

if args.log_wandb:
    logger.info(f"Instantiating wandb config project=<{args.project}> entity=<{args.entity}>")
    logger.info(f"Instanting run for <{args.wandb_name}>")

        # initiate wandb
    initiate_wandb(args)
else:
    logger.info("Wandb logging is disabled")





print('data processing:')
args.features_path=  f'{args.data_root}/{args.dataset}/features/'
training_dataloader = DataGenerator(data_root=args.data_root, split=args.split,
                                    dataset=args.dataset, mode='train',
                                    args=args, usebatch=args.usebatch, len_seg_max=args.len_seg_max,
                                    features_path=args.features_path, feature_type=args.feature_type, feature_mode=args.feature_mode,action = 'dav3')
trainloader = torch.utils.data.DataLoader(training_dataloader,
                                          batch_size=args.bs,
                                          shuffle=True,
                                          num_workers=4 ,
                                          pin_memory=True)  
# dataloaders testing                    
testing_dataloader = DataGenerator(data_root=args.data_root, split=args.split,
                                    dataset=args.dataset, mode='val',args=args,
                                    features_path=args.features_path, feature_type=args.feature_type, feature_mode=args.feature_mode, action = 'dav3')
testloader = torch.utils.data.DataLoader(testing_dataloader,
                                         batch_size=1,
                                         shuffle=False,
                                         num_workers=4,
                                         pin_memory=True)             
print('starting the training:')
trainer = Trainer(args)
trainer.train(args, device=device, trainloader=trainloader, testloader=testloader, testing_dataloader=testing_dataloader)

print('end time:')
print(datetime.datetime.now(pytz.timezone('US/Eastern')).isoformat())

# if __name__ == '__main__' :
    
#     dataset= '50salads'
#     data_root= '/home/ahmed/Ahmed_data/UVAST/UVAST/data'
#     split = 1
#     dataset=DataGenerator(data_root=data_root,
#                  split=1,
#                  dataset=dataset,
#                  mode='train',
#                  transform=None,
#                  usebatch=False,
#                  args=args,
#                  len_seg_max=100,
#                  features_path=None,
#                  feature_type=".npy",
#                  feature_mode="feature")