import torch.nn as nn
import torch
import os
import numpy as np
import scipy.io as sio
from torch.utils.data import Dataset, DataLoader
import platform
import glob
from argparse import ArgumentParser
from test_function import test_implement_MRI
from ACIU import ACIUNet, FFT_Mask_ForBack

###########################################################################################
# parameter
parser = ArgumentParser(description='ACIU-Net')
parser.add_argument('--start_epoch', type=int, default=0, help='epoch number of start training')
parser.add_argument('--end_epoch', type=int, default=300, help='epoch number of end training')
parser.add_argument('--layer_num', type=int, default=9, help='phase number of ACIUNet')
parser.add_argument('--learning_rate', type=float, default=1e-4, help='learning rate')
parser.add_argument('--cs_ratio', type=int, default=20, help='from {10, 20, 30, 40, 50}')
parser.add_argument('--gpu_list', type=str, default='0', help='gpu index')
parser.add_argument('--matrix_dir', type=str, default='sampling_matrix', help='sampling matrix directory')
parser.add_argument('--model_dir', type=str, default='model', help='trained or pre-trained model directory')
parser.add_argument('--data_dir', type=str, default='data', help='training data directory')
parser.add_argument('--log_dir', type=str, default='log', help='log directory')
parser.add_argument('--net_name', type=str, default='ACIU-Net', help='name of net')
parser.add_argument('--num-layers', type=int, default=8, help='C,8')
parser.add_argument('--result_dir', type=str, default='result', help='result directory')
parser.add_argument('--test_name', type=str, default='BrainImages_test', help='name of test set')
parser.add_argument('--run_mode', type=str, default='train', help='train、test')
parser.add_argument('--print_flag', type=int, default=1, help='print parameter number 1 or 0')
args = parser.parse_args()

#########################################################################################
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
def seed_everything(seed):

    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
seed_everything(42)
###########################################################################################
# parameter
nrtrain = 800   # number of training blocks
batch_size = 1
start_epoch = args.start_epoch
end_epoch = args.end_epoch
learning_rate = args.learning_rate
layer_num = args.layer_num

cs_ratio = args.cs_ratio
gpu_list = args.gpu_list
Training_data_Name = 'Training_BrainImages_256x256_100.mat'
# define save dir
model_dir = "./%s/%s_layer_%d_denselayer_%d_ratio_%d_lr_%f" % (args.model_dir, args.net_name,args.layer_num, args.num_layers, args.cs_ratio, args.learning_rate)
test_dir = os.path.join(args.data_dir, args.test_name)   # test image dir
filepaths = glob.glob(test_dir + '/*.png')
output_file_name = "./%s/%s_layer_%d_denselayer_%d_ratio_%d_lr_%f.txt" % (args.log_dir, args.net_name, args.layer_num, args.num_layers, args.cs_ratio, args.learning_rate)
#########################################################################################
# Load CS Sampling Matrix: phi
Phi_data_Name = './%s/mask_%d.mat' % (args.matrix_dir, args.cs_ratio)
Phi_data = sio.loadmat(Phi_data_Name)
mask_matrix = Phi_data['mask_matrix']
mask_matrix = torch.from_numpy(mask_matrix).type(torch.FloatTensor)
mask = torch.unsqueeze(mask_matrix, 2)
mask = torch.cat([mask, mask], 2)
mask = mask.to(device)
Training_data = sio.loadmat('./%s/%s' % (args.data_dir, Training_data_Name))
Training_labels = Training_data['labels']

# initial test file
result_dir = os.path.join(args.result_dir, args.test_name)
result_dir = result_dir+'_'+args.net_name+'_ratio_'+ str(args.cs_ratio)+'_epoch_'+str(args.end_epoch)+'/'
if not os.path.exists(result_dir):
    os.makedirs(result_dir)
###################################################################################
# model
model = ACIUNet(layer_num)
model = nn.DataParallel(model)
model = model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
# optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
###################################################################################
if args.print_flag:  # print networks parameter number
    print("Total number of paramerters in networks is {}  ".format(sum(x.numel() for x in model.parameters())))
####################################################################################
class RandomDataset(Dataset):
    def __init__(self, data, length):
        self.data = data
        self.len = length
    def __getitem__(self, index):
        return torch.Tensor(self.data[index, :]).float()
    def __len__(self):
        return self.len
#####################################################################################
if (platform.system() =="Windows"):
    rand_loader = DataLoader(dataset=RandomDataset(Training_labels, nrtrain), batch_size=batch_size, num_workers=0,
                             shuffle=True)
else:
    rand_loader = DataLoader(dataset=RandomDataset(Training_labels, nrtrain), batch_size=batch_size, num_workers=4,
                             shuffle=True)
#######################################################################################
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
if args.start_epoch > 0:   # train stop and restart
    pre_model_dir = model_dir
    model.load_state_dict(torch.load('./%s/net_params_%d.pkl' % (pre_model_dir, args.start_epoch)))
#########################################################################################
if args.run_mode == 'train':
    # Training loop
    for epoch_i in range(args.start_epoch+1, args.end_epoch+1):
        model = model.train()

        loss_list = []
        for data in rand_loader:


            batch_x = data
            batch_x = batch_x.to(device)
            batch_x = batch_x.view(batch_x.shape[0], 1, batch_x.shape[1], batch_x.shape[2])
            PhiTb = FFT_Mask_ForBack()(batch_x, mask)
            x_output = model(PhiTb, mask)

            loss_all  = torch.mean(torch.abs(x_output - batch_x))
            optimizer.zero_grad()
            loss_all.backward()
            optimizer.step()
            loss_list.append(loss_all.item())


        output_data = "[%02d/%02d] Loss: %.6f" % \
                      (epoch_i, args.end_epoch, np.mean(loss_list),
                       )
        print(output_data)

               # Load pre-trained model with epoch number
        model = model.eval()

        # save result
        file_data = [epoch_i, np.mean(loss_list)]
        output_file = open(output_file_name, 'a')
        for fp in file_data:   # write data in txt
            output_file.write(str(fp))
            output_file.write(',')
        output_file.write('\n')    # line feed
        output_file.close()

        # save model in every epoch
        torch.save(model.state_dict(), "./%s/net_params_%d.pkl" % (model_dir, epoch_i))  # save only the parameters

elif args.run_mode=='test':
    model.load_state_dict(torch.load('./%s/net_params_%d.pkl' % (model_dir, args.end_epoch)))
    # Load pre-trained model with epoch number
    model = model.eval()
    PSNR_mean, SSIM_mean, RMSE_mean = test_implement_MRI(filepaths, model, args.cs_ratio,mask,args.test_name,args.end_epoch,result_dir,
                                              args.run_mode,device)

