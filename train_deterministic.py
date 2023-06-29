# The code for this paper is modified from SGNet paper code:
# https://github.com/ChuhuaW/SGNet.pytorch

import os
import os.path as osp
import torch
from torch import nn, optim
from lib.utils.data_utils import build_data_loader
from configs.pie import parse_sgnet_args as parse_args
from lib.models.Trajnet import Trajnet
from lib.losses import rmse_loss
from lib.utils.train_val_test import train, val, test
from torch.utils.tensorboard import SummaryWriter

def main(args):
    this_dir = osp.dirname(__file__)
    logs_dir = osp.join(this_dir,"runs",args.version_name)
    if not osp.isdir(logs_dir):
        os.makedirs(logs_dir)
    writer = SummaryWriter(log_dir=logs_dir)
    val_save_dir = osp.join(this_dir, 'checkpoints', args.version_name, 'val')
    test_save_dir = osp.join(this_dir, 'checkpoints', args.version_name, 'test')
    if not osp.isdir(val_save_dir):
        os.makedirs(val_save_dir)
    if not osp.isdir(test_save_dir):
        os.makedirs(test_save_dir)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = Trajnet(args, device)
    model = nn.DataParallel(model)
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.2, patience=args.patience,
                                                            min_lr=1e-10, verbose=1)
    if osp.isfile(args.checkpoint):    
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        args.start_epoch += checkpoint['epoch']      

    criterion = rmse_loss().to(device)
    
    train_gen, scaler_sp = build_data_loader(args, 'train')
    val_gen,_ = build_data_loader(args, 'val', scaler_sp = scaler_sp)
    test_gen,_ = build_data_loader(args, 'test', scaler_sp = scaler_sp)
    print("Number of validation samples:", val_gen.__len__())
    print("Number of test samples:", test_gen.__len__())

    min_loss = 1e6
    for epoch in range(args.start_epoch, args.epochs):

        total_train_loss = train(model, train_gen, criterion, optimizer, device, epoch, writer, args)
        print('Train Epoch: {} \t Total: {:.4f}'.format(
                epoch, total_train_loss))
        val_loss, MSE_15, MSE_05, MSE_10, FMSE_15, FMSE_05, FMSE_10, FIOU_15, FIOU_05, FIOU_10, CMSE_15, CMSE_05, CMSE_10,\
              CFMSE_15, CFMSE_05, CFMSE_10 = val(model, val_gen, criterion, device, epoch, writer, args)
        lr_scheduler.step(val_loss)
        print("Validation Loss: {:.4f}".format(val_loss))
        print("MSE_05: %4f;  MSE_10: %4f;  MSE_15: %4f\n" % (MSE_05, MSE_10, MSE_15))
        if val_loss < min_loss:
            try:
                os.remove(best_model_metric)
            except:
                pass

            min_loss = val_loss
            with open(os.path.join(val_save_dir, 'metric.txt'),"w") as f:
                f.write("%4f\t%4f\t%4f\t%4f\t%4f\t%4f\t%4f\t%4f\t%4f\t%4f\t%4f\t%4f\t%4f\t%4f\t%4f\n" % (MSE_05, MSE_10, MSE_15, FMSE_05, FMSE_10, FMSE_15, FIOU_05, FIOU_10, FIOU_15, CFMSE_05, CFMSE_10, CFMSE_15, CMSE_05, CMSE_10, CMSE_15))

            saved_model_metric_name = 'metric_epoch_' + str(format(epoch,'03')) + '_loss_%.4f'%min_loss + '.pth'

            print("Saving checkpoints: " + saved_model_metric_name)
            if not os.path.isdir(val_save_dir):
                os.mkdir(val_save_dir)
            save_dict = {   'epoch': epoch,
                            'model_state_dict': model.module.state_dict(), 
                            'optimizer_state_dict': optimizer.state_dict()}
            torch.save(save_dict, os.path.join(val_save_dir, saved_model_metric_name))


            best_model_metric = os.path.join(val_save_dir, saved_model_metric_name)

        # test
            test_loss, MSE_15, MSE_05, MSE_10, FMSE_15, FMSE_05, FMSE_10, FIOU_15, FIOU_05, FIOU_10, CMSE_15, CMSE_05, CMSE_10, CFMSE_15, CFMSE_05, CFMSE_10 = test(model, test_gen, criterion, device, epoch, writer, args)
            print("Test Loss: {:.4f}".format(test_loss))
            print("MSE_05: %4f;  MSE_10: %4f;  MSE_15: %4f\n" % (MSE_05, MSE_10, MSE_15))
            with open(os.path.join(test_save_dir, 'metric.txt'),"w") as f:
                f.write("%4f\t%4f\t%4f\t%4f\t%4f\t%4f\t%4f\t%4f\t%4f\t%4f\t%4f\t%4f\t%4f\t%4f\t%4f\n" % (MSE_05, MSE_10, MSE_15, FMSE_05, FMSE_10, FMSE_15, FIOU_05, FIOU_10, FIOU_15, CFMSE_05, CFMSE_10, CFMSE_15, CMSE_05, CMSE_10, CMSE_15))

        # save checkpoints if test MSE decreases
    writer.close() 


if __name__ == '__main__':
    main(parse_args())