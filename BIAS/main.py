import time, os, torch, argparse, warnings, glob

from dataLoader import train_loader, val_loader
from utils.tools import *
from bias import bias

def main():
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
    args = parser.parse_args()
    
    # Data loader
    args = init_args(args)

    loader = train_loader(trialFileName = args.trainTrial, \
                          audioPath      = os.path.join(args.audioPathFolder, 'train'), \
                          visualPath     = os.path.join(args.visualPathFolder, 'train'), \
                          visualPathBody     = os.path.join(args.visualPathBodyFolder, 'train'), \
                          **vars(args))
    trainLoader = torch.utils.data.DataLoader(loader, batch_size = 1, shuffle = True, num_workers = args.nDataLoaderThread)

    loader = val_loader(trialFileName = args.evalTrial, \
                        audioPath     = os.path.join(args.audioPathFolder, args.evalDataType), \
                        visualPath    = os.path.join(args.visualPathFolder, args.evalDataType), \
                        visualPathBody     = os.path.join(args.visualPathBodyFolder, args.evalDataType), \
                        **vars(args))

    valLoader = torch.utils.data.DataLoader(loader, batch_size = 1, shuffle = False, num_workers = args.nDataLoaderThread)

    if args.evaluation == True:
        download_pretrain_model()
        s = bias(**vars(args))
        s.loadParameters('pretrained_BIAS.model')
        mAP = s.evaluate_network(loader = valLoader, **vars(args))
        print("mAP %2.2f%%"%(mAP))
        quit()

    modelfiles = glob.glob('%s/model_0*.model'%args.modelSavePath)
    modelfiles.sort()  
    if len(modelfiles) >= 1:
        print("Model %s loaded from previous state!"%modelfiles[-1])
        epoch = int(os.path.splitext(os.path.basename(modelfiles[-1]))[0][6:]) + 1
        s = bias(epoch = epoch, **vars(args))
        s.loadParameters(modelfiles[-1])
    else:
        epoch = 1
        s = bias(epoch = epoch, **vars(args))

    mAPs = []
    scoreFile = open(args.scoreSavePath, "a+")

    while(1):        
        loss, lr = s.train_network(epoch = epoch, loader = trainLoader, **vars(args))
        
        if epoch % args.testInterval == 0:        
            s.saveParameters(args.modelSavePath + "/model_%04d.model"%epoch)
            mAPs.append(s.evaluate_network(epoch = epoch, loader = valLoader, **vars(args)))
            print(time.strftime("%Y-%m-%d %H:%M:%S"), "%d epoch, mAP %2.2f%%, bestmAP %2.2f%%"%(epoch, mAPs[-1], max(mAPs)))
            scoreFile.write("%d epoch, LR %f, LOSS %f, mAP %2.2f%%, bestmAP %2.2f%%\n"%(epoch, lr, loss, mAPs[-1], max(mAPs)))
            scoreFile.flush()

        if epoch >= args.maxEpoch:
            quit()

        epoch += 1

if __name__ == '__main__':
    main()
