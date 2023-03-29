import time, os, torch, argparse, warnings, glob

from dataLoader import train_loader, val_loader
from utils.tools import *
from bias import bias
import torch.nn as nn

def main():
    # The structure of this code is learnt from https://github.com/clovaai/voxceleb_trainer
    warnings.filterwarnings("ignore")

    parser = argparse.ArgumentParser(description = "BIAS Training")
    # Training details
    parser.add_argument('--lr',           type=float, default=0.0001,help='Learning rate')
    parser.add_argument('--lrDecay',      type=float, default=0.95,  help='Learning rate decay rate')
    parser.add_argument('--maxEpoch',     type=int,   default=25,    help='Maximum number of epochs')
    parser.add_argument('--testInterval', type=int,   default=1,     help='Test and save every [testInterval] epochs')
    parser.add_argument('--batchSize',    type=int,   default=2500,  help='Dynamic batch size, default is 2500 frames, other batchsize (such as 1500) will not affect the performance')
    
    parser.add_argument('--nDataLoaderThread', type=int, default=1,  help='Number of loader threads')
    # Data path
    parser.add_argument('--dataPath',  type=str, default="WASD/", help='Save path of dataset')
    parser.add_argument('--savePath',     type=str, default="exps/wasd")
    # Data selection
    parser.add_argument('--evalDataType', type=str, default="val", help='Only for AVA, to choose the dataset for evaluation, val or test')
    # For download dataset only, for evaluation only
    parser.add_argument('--evaluation',      dest='evaluation', action='store_true', help='Only do evaluation by using pretrained model')
   
    parser.add_argument('--changeAudio',     dest='changeAudio', action='store_true', help='')
    parser.add_argument('--changeFace',      dest='changeFace', action='store_true', help='')
    parser.add_argument('--changeBody',      dest='changeBody', action='store_true', help='')
    parser.add_argument('--outputFile',      type=str, default="val_res.csv") 
    parser.add_argument('--inputModel',      type=str, default="") 
    
    args = parser.parse_args()
    # Data loader
    args = init_args(args)

    loader = val_loader(trialFileName = args.evalTrial, \
                        audioPath     = os.path.join(args.audioPathFolder, args.evalDataType), \
                        visualPath    = os.path.join(args.visualPathFolder, args.evalDataType), \
                        visualPathBody     = os.path.join(args.visualPathBodyFolder, args.evalDataType), \
                        **vars(args))

    valLoader = torch.utils.data.DataLoader(loader, batch_size = 1, shuffle = False, num_workers = args.nDataLoaderThread)

    if args.evaluation == True:
        s = bias(**vars(args))
        kwargs = vars(args)
        s.loadParameters(kwargs["inputModel"])
        print("Loaded Model:", kwargs["inputModel"])
        mAP = s.evaluate_network(loader = valLoader, **vars(args))
        print("mAP %2.2f%%\n"%(mAP))
        quit()

if __name__ == '__main__':
    main()
