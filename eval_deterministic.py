import os
import os.path as osp
import torch
from torch import nn
from lib.models.Trajnet import Trajnet

from configs.pie import parse_sgnet_args as parse_args
from lib.losses import rmse_loss
from lib.utils.train_val_test import test
from lib.utils.data_utils import build_data_loader
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()

def main(args):

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Trajnet(args, device)

    if osp.isfile(args.checkpoint):
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'],strict=True)
        print("read checkpoint")
    model = nn.DataParallel(model)
    model = model.to(device)

    criterion = rmse_loss().to(device)
    # to have the normalization values for inputs.
    train_gen, scaler_sp = build_data_loader(args, 'train')
    test_gen,_ = build_data_loader(args, 'test', scaler_sp = scaler_sp)
    print("Number of test samples:", test_gen.__len__())

    # test
    epoch = 0

    test_loss, MSE_15, MSE_05, MSE_10, FMSE_15, FMSE_05, FMSE_10, FIOU_15, FIOU_05, FIOU_10, CMSE_15, CMSE_05, CMSE_10, CFMSE_15, CFMSE_05,\
    CFMSE_10  = test(model, test_gen, criterion, device, epoch, writer, args)
    print("MSE_05: %4f;  MSE_10: %4f;  MSE_15: %4f;   FMSE: %4f;   FIOU: %4f\n" % (MSE_05, MSE_10, MSE_15, FMSE_15, FIOU_15))
    print("CFMSE: %4f;   CMSE: %4f;  \n" % (CFMSE_15, CMSE_15))

if __name__ == '__main__':
    main(parse_args())
