from pytorch_grad_cam.utils.image import show_cam_on_image

import os, torch, argparse, warnings

from dataLoader import val_loader
from utils.tools import *
from bias import bias
import torch.nn.functional as nnf
import numpy as np
import cv2
from tqdm import tqdm


def top_n_chanels(se_vetor_sorted):

    l = list(se_vetor_sorted.cpu())
    array = numpy.array(l)
    mean, std = numpy.mean(array, axis=0), numpy.std(array, axis=0)
    # Alternative to get top 10%, X = u + Zo, u = mean and o = std
    # For percentil2:
    # 90%, Z = 1.282
    # 95%, Z = 1.645
    Z = 1.282

    threshold = mean + Z*std
    list_channels = [e for e in l if e > threshold ]
    r = len(list_channels)

    return r

def draw_se_images(se_vetor, feat, output_dir, list_img_path):

    se_vetor = torch.squeeze(se_vetor)
    se_vetor_sorted, indices = torch.sort(se_vetor, descending=True)

    # Get top %
    numb_chanels = top_n_chanels(se_vetor_sorted)

    indices = indices[:numb_chanels]
    best_feat = torch.index_select(feat, 1, indices)

    scale_best_imgs = nnf.interpolate(best_feat, size=(112, 112), mode='bicubic', align_corners=False)
    scale_best_imgs = scale_best_imgs.squeeze()

    img_i = 0
    scale_best_imgs = scale_best_imgs.cpu().detach().numpy()
    os.makedirs(output_dir, exist_ok=True)

    for img_path in tqdm(list_img_path):
        img_path = img_path[0]
        rgb_img = cv2.imread(img_path, 1)[:, :, ::-1]
        rgb_img = cv2.resize(rgb_img, (112,112), interpolation = cv2.INTER_AREA)
        rgb_img = np.float32(rgb_img) / 255

        heatmap = scale_best_imgs[img_i][0]
        for i in range(1, numb_chanels):
            heatmap += scale_best_imgs[img_i][i]

        heatmap = heatmap/numb_chanels

        cam_image = show_cam_on_image(rgb_img, heatmap, use_rgb=True)
        cam_image = cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR)

        cv2.imwrite('{}/visual_{}.jpg'.format(output_dir, str(img_i)), cam_image)
        img_i += 1


if __name__ == '__main__':
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

    s = bias(**vars(args))
    s.loadParameters('pretrain_BIAS_Visual.model')

    se_face_feat, se_face_vetor, face_path, se_body_feat, se_body_vetor, body_path = s.evaluate_network_se(loader = valLoader, **vars(args))
    output_face_dir = "visual_images/face"
    output_body_dir = "visual_images/body"

    os.makedirs(output_face_dir, exist_ok=True)
    os.makedirs(output_body_dir, exist_ok=True)

    print("Face SE Images")
    draw_se_images(se_face_vetor, se_face_feat, output_face_dir, face_path)
    print("Body SE Images")
    draw_se_images(se_body_vetor, se_body_feat, output_body_dir, body_path)
