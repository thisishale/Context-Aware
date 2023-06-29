from tqdm import tqdm
import torch
from lib.utils.eval_utils import eval_jaad_pie

def train(model, train_gen, criterion, optimizer, device, epoch, writer, args):
    model.train() # Sets the module in training mode.
    total_loss = 0
    loader = tqdm(train_gen, total=len(train_gen))
    with torch.set_grad_enabled(True):
        for batch_idx, data in enumerate(loader):
            
            batch_size = data['input_x'].shape[0]
            input_traj = data['input_x'].to(device)
            input_speed = data['input_speed'].to(device)
            target_traj = data['target_y'].to(device)
            dec_traj = model(inputs=[input_traj, input_speed], targets = target_traj, mask = None)
            dec_loss = criterion(dec_traj, target_traj)

            total_loss += (dec_loss.item()* batch_size) 
            optimizer.zero_grad()
            dec_loss.backward()
            optimizer.step()
        
    total_loss/=len(train_gen.dataset)
    writer.add_scalar("train/total_loss", total_loss, epoch)
    return total_loss

def val(model, val_gen, criterion, device, epoch, writer, args):
    total_loss = 0
    MSE_15, MSE_05, MSE_10, FMSE_15, FMSE_05, FMSE_10, FIOU_15,\
    FIOU_05, FIOU_10, CMSE_15, CMSE_05, CMSE_10, CFMSE_15, CFMSE_05\
    , CFMSE_10= 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    model.eval()
    loader = tqdm(val_gen, total=len(val_gen))
    with torch.set_grad_enabled(False):
        for batch_idx, data in enumerate(loader):
            batch_size = data['input_x'].shape[0]
            input_traj = data['input_x'].to(device)
            input_speed = data['input_speed'].to(device)
            target_traj = data['target_y'].to(device)
            dec_traj  = model(inputs=[input_traj, input_speed], targets = target_traj, mask = None, training=False)
            dec_loss = criterion(dec_traj, target_traj)            

            total_loss += (dec_loss.item()* batch_size) 
            
            input_traj_np = input_traj.to('cpu').numpy()
            target_traj_np = target_traj.to('cpu').numpy()
            dec_traj = dec_traj.to('cpu').numpy()
            batch_MSE_15, batch_MSE_05, batch_MSE_10, batch_FMSE_15, batch_FMSE_05, batch_FMSE_10,\
            batch_CMSE_15, batch_CMSE_05, batch_CMSE_10, batch_CFMSE_15, batch_CFMSE_05, batch_CFMSE_10,\
             batch_FIOU_15, batch_FIOU_05, batch_FIOU_10 = eval_jaad_pie(input_traj_np, target_traj_np, dec_traj)
            MSE_15 += batch_MSE_15
            MSE_05 += batch_MSE_05
            MSE_10 += batch_MSE_10
            FMSE_15 += batch_FMSE_15
            FMSE_05 += batch_FMSE_05
            FMSE_10 += batch_FMSE_10
            CMSE_15 += batch_CMSE_15
            CMSE_05 += batch_CMSE_05
            CMSE_10 += batch_CMSE_10
            CFMSE_15 += batch_CFMSE_15
            CFMSE_05 += batch_CFMSE_05
            CFMSE_10 += batch_CFMSE_10
            FIOU_15 += batch_FIOU_15
            FIOU_05 += batch_FIOU_05
            FIOU_10 += batch_FIOU_10
            

    
    MSE_15 /= len(val_gen.dataset)
    MSE_05 /= len(val_gen.dataset)
    MSE_10 /= len(val_gen.dataset)
    FMSE_15 /= len(val_gen.dataset)
    FMSE_05 /= len(val_gen.dataset)
    FMSE_10 /= len(val_gen.dataset)
    FIOU_15 /= len(val_gen.dataset)
    FIOU_05 /= len(val_gen.dataset)
    FIOU_10 /= len(val_gen.dataset)
    CMSE_15 /= len(val_gen.dataset)
    CMSE_05 /= len(val_gen.dataset)
    CMSE_10 /= len(val_gen.dataset)
    CFMSE_15 /= len(val_gen.dataset)
    CFMSE_05 /= len(val_gen.dataset)
    CFMSE_10 /= len(val_gen.dataset)

    val_loss = total_loss/len(val_gen.dataset)
    writer.add_scalar("Val/val_loss", val_loss, epoch)
    writer.add_scalar("Val/MSE_15", MSE_15, epoch)
    writer.add_scalar("Val/MSE_05", MSE_05, epoch)
    writer.add_scalar("Val/MSE_10", MSE_10, epoch)
    writer.add_scalar("Val/FMSE_15", FMSE_15, epoch)
    writer.add_scalar("Val/FMSE_05", FMSE_05, epoch)
    writer.add_scalar("Val/FMSE_10", FMSE_10, epoch)
    writer.add_scalar("Val/FIOU_15", FIOU_15, epoch)
    writer.add_scalar("Val/FIOU_05", FIOU_05, epoch)
    writer.add_scalar("Val/FIOU_10", FIOU_10, epoch)
    writer.add_scalar("Val/CMSE_15", CMSE_15, epoch)
    writer.add_scalar("Val/CMSE_05", CMSE_05, epoch)
    writer.add_scalar("Val/CMSE_10", CMSE_10, epoch)
    writer.add_scalar("Val/CFMSE_15", CFMSE_15, epoch)
    writer.add_scalar("Val/CFMSE_05", CFMSE_05, epoch)
    writer.add_scalar("Val/CFMSE_10", CFMSE_10, epoch)
    return val_loss, MSE_15, MSE_05, MSE_10, FMSE_15, FMSE_05, FMSE_10, FIOU_15, FIOU_05, FIOU_10, CMSE_15, CMSE_05, CMSE_10, CFMSE_15,\
          CFMSE_05, CFMSE_10 

def test(model, test_gen, criterion, device, epoch, writer, args):
    total_loss = 0
    MSE_15, MSE_05, MSE_10, FMSE_15, FMSE_05, FMSE_10, FIOU_15,\
    FIOU_05, FIOU_10, CMSE_15, CMSE_05, CMSE_10, CFMSE_15, CFMSE_05\
    , CFMSE_10= 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    model.eval()
    loader = tqdm(test_gen, total=len(test_gen))
    with torch.set_grad_enabled(False):
        for batch_idx, data in enumerate(loader):     
            batch_size = data['input_x'].shape[0]
            input_traj = data['input_x'].to(device)
            input_speed = data['input_speed'].to(device)
            target_traj = data['target_y'].to(device)
            dec_traj  = model(inputs=[input_traj, input_speed], targets = target_traj, mask = None, training=False, loop=1)
            dec_loss = criterion(dec_traj, target_traj)            

            total_loss += (dec_loss.item()* batch_size)  

            input_traj_np = input_traj.to('cpu').numpy()
            target_traj_np = target_traj.to('cpu').numpy()
            dec_traj = dec_traj.to('cpu').numpy()
            batch_MSE_15, batch_MSE_05, batch_MSE_10, batch_FMSE_15, batch_FMSE_05, batch_FMSE_10,\
            batch_CMSE_15, batch_CMSE_05, batch_CMSE_10, batch_CFMSE_15, batch_CFMSE_05, batch_CFMSE_10,\
            batch_FIOU_15, batch_FIOU_05, batch_FIOU_10 = eval_jaad_pie(input_traj_np, target_traj_np, dec_traj)
            MSE_15 += batch_MSE_15
            MSE_05 += batch_MSE_05
            MSE_10 += batch_MSE_10
            FMSE_15 += batch_FMSE_15
            FMSE_05 += batch_FMSE_05
            FMSE_10 += batch_FMSE_10
            CMSE_15 += batch_CMSE_15
            CMSE_05 += batch_CMSE_05
            CMSE_10 += batch_CMSE_10
            CFMSE_15 += batch_CFMSE_15
            CFMSE_05 += batch_CFMSE_05
            CFMSE_10 += batch_CFMSE_10
            FIOU_15 += batch_FIOU_15
            FIOU_05 += batch_FIOU_05
            FIOU_10 += batch_FIOU_10

            
    
    MSE_15 /= len(test_gen.dataset)
    MSE_05 /= len(test_gen.dataset)
    MSE_10 /= len(test_gen.dataset)
    FMSE_15 /= len(test_gen.dataset)
    FMSE_05 /= len(test_gen.dataset)
    FMSE_10 /= len(test_gen.dataset)
    FIOU_15 /= len(test_gen.dataset)
    FIOU_05 /= len(test_gen.dataset)
    FIOU_10 /= len(test_gen.dataset)
    CMSE_15 /= len(test_gen.dataset)
    CMSE_05 /= len(test_gen.dataset)
    CMSE_10 /= len(test_gen.dataset)
    CFMSE_15 /= len(test_gen.dataset)
    CFMSE_05 /= len(test_gen.dataset)
    CFMSE_10 /= len(test_gen.dataset)

    total_loss = total_loss/len(test_gen.dataset)
    writer.add_scalar("test/total_loss", total_loss, epoch)
    writer.add_scalar("test/MSE_15", MSE_15, epoch)
    writer.add_scalar("test/MSE_05", MSE_05, epoch)
    writer.add_scalar("test/MSE_10", MSE_10, epoch)
    writer.add_scalar("test/FMSE_15", FMSE_15, epoch)
    writer.add_scalar("test/FMSE_05", FMSE_05, epoch)
    writer.add_scalar("test/FMSE_10", FMSE_10, epoch)
    writer.add_scalar("test/FIOU_15", FIOU_15, epoch)
    writer.add_scalar("test/FIOU_05", FIOU_05, epoch)
    writer.add_scalar("test/FIOU_10", FIOU_10, epoch)
    writer.add_scalar("test/CMSE_15", CMSE_15, epoch)
    writer.add_scalar("test/CMSE_05", CMSE_05, epoch)
    writer.add_scalar("test/CMSE_10", CMSE_10, epoch)
    writer.add_scalar("test/CFMSE_15", CFMSE_15, epoch)
    writer.add_scalar("test/CFMSE_05", CFMSE_05, epoch)
    writer.add_scalar("test/CFMSE_10", CFMSE_10, epoch)
    
    return total_loss, MSE_15, MSE_05, MSE_10, FMSE_15, FMSE_05, FMSE_10, FIOU_15, FIOU_05, FIOU_10, CMSE_15, CMSE_05, CMSE_10, CFMSE_15, CFMSE_05, CFMSE_10 



