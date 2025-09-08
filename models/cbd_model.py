""" 
CBD (Controllable Blind Decomposition) model for Task II.A , real-scenario deraining in driving.
Based on Restormer architecture with modifications for controllable multi-degradation removal.

Sample usage:
Optional visualization:
python -m visdom.server

Train:
python train.py --dataroot ./datasets/raina --name task2a --model cbd --dataset_mode raina

For all 6 testing cases, run following one by one:
python test.py --dataroot ./datasets/raina --name task2a --model cbd --dataset_mode raina --test_input B 
python test.py --dataroot ./datasets/raina --name task2a --model cbd --dataset_mode raina --test_input BC
python test.py --dataroot ./datasets/raina --name task2a --model cbd --dataset_mode raina --test_input BD --haze_intensity 0
python test.py --dataroot ./datasets/raina --name task2a --model cbd --dataset_mode raina --test_input BD --haze_intensity 2
python test.py --dataroot ./datasets/raina --name task2a --model cbd --dataset_mode raina --test_input BDE --haze_intensity 1
python test.py --dataroot ./datasets/raina --name task2a --model cbd --dataset_mode raina --test_input BCDE --haze_intensity 1  
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_model import BaseModel
from . import networks
import random
from .losses import VGGLoss
from numpy import *
import itertools
import cv2
import numpy as np
from random import choice
from skimage.metrics import peak_signal_noise_ratio as psnr


class CBDModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """  Configures options specific for CBD model.
        """
        parser.add_argument('--lambda_Ln', type=float, default=50.0, help='weight for L1/L2 loss') # 30.0, 45.0
        parser.add_argument('--lambda_VGG', type=float, default=15.0, help='weight for VGG loss')
        parser.add_argument('--lambda_BCE', type=float, default=1.0, help='weight for BCE loss')
        parser.add_argument('--lambda_Dice', type=float, default=2.0, help='weight for Dice loss')
        parser.add_argument('--lambda_Huber', type=float, default=10.0, help='weight for Huber loss')
        parser.add_argument('--test_input', type=str, default='B', help='test images, B = rain streak,'
                                                                          ' C = snow, D = haze, E = raindrop.')
        parser.add_argument('--max_domain', type=int, default=4, help='max number of source components.')
        parser.add_argument('--prob1', type=float, default=1.0, help='probability of adding rain streak (A)')
        parser.add_argument('--prob2', type=float, default=0.5, help='probability of adding other components')
        parser.add_argument('--haze_intensity', type=int, default=1, help='intensity of haze, only matters for testing. '
                                                                          '0: light, 1: moderate, 2: heavy.')
        opt, _ = parser.parse_known_args()
        return parser

    def __init__(self, opt):
        # Specify the training losses you want to print out.
        # The training/test scripts will call <BaseModel.get_current_losses>
        BaseModel.__init__(self, opt)
        self.loss_names = ['LnR0','VGGR0','LnR1','VGGR1','LnR2','Huber','Lnmask','VGGmask','BCE','PSNR0','PSNR1','CE','Dice']
        # , 'fake_input', 'fake_MixI', 'fake_MixA'
        self.visual_names = ['fake_A', 'real_A', 'fake_B', 'real_B', 'fake_H', 'real_H', 'fake_I', 'real_I2', \
             'fake_input','real_input', 'input_noised', 'real_noise']
        self.model_names = ['R']

        # Define networks
        self.netR = networks.define_R(opt.input_nc, opt.output_nc, opt.ngf, 'head', opt.normG,
                                       not opt.no_dropout, opt.init_type, opt.init_gain, opt.no_antialias,
                                       opt.no_antialias_up, self.gpu_ids, opt)


        self.label = torch.zeros(self.opt.max_domain).to(self.device)

        self.accuracy = 0
        self.Seg_CE = 0
        self.psnr = 0
        self.all_count = 0
        self.total_iou = 0
        self.total_pixel_acc = 0
        self.Seg_Dice = 0
        self.Seg_Focal = 0
        self.width = self.opt.crop_size

        self.scaler = torch.cuda.amp.GradScaler()  # initialize a GradScaler object

        if self.isTrain:
            # define loss functions
            self.criterionL1 = torch.nn.L1Loss().to(self.device)
            self.criterionL2 = torch.nn.MSELoss().to(self.device)
            self.criterionVGG = VGGLoss(opt).to(self.device)
            #self.criterionBCE = torch.nn.BCELoss().to(self.device)
            self.criterionBCE = torch.nn.BCEWithLogitsLoss().to(self.device)
            self.criterionPSNR = self.PSNRLoss().to(self.device)
            self.criterionCE = torch.nn.CrossEntropyLoss().to(self.device)
            self.criterionDice = DiceLoss(64).to(self.device)
            self.criterionHuber = torch.nn.SmoothL1Loss().to(self.device)
            self.optimizer_R = torch.optim.AdamW(self.netR.parameters(), lr=opt.lr, weight_decay=1e-4)
            self.optimizers.append(self.optimizer_R)
    
    class PSNRLoss(nn.Module):
        def __init__(self):
            super().__init__()
            self.L2 = torch.nn.MSELoss()
            
        def forward(self, x, y):
            return 6-10*torch.log10(self.L2(x, y))

    def data_dependent_initialize(self, data):
        self.set_input(data)
        bs_per_gpu = self.real_A.size(0) // max(len(self.opt.gpu_ids), 1)
        self.real_A = self.real_A[:bs_per_gpu]
        self.real_B = self.real_B[:bs_per_gpu]
        self.real_C = self.real_C[:bs_per_gpu]
        self.real_D = self.real_D[:bs_per_gpu]
        self.real_E = self.real_E[:bs_per_gpu]
        self.real_E2 = self.real_E2[:bs_per_gpu]
        self.real_H = self.real_H[:bs_per_gpu]
        self.real_I = self.real_I[:bs_per_gpu]
        self.forward()
        if self.opt.isTrain:
            self.compute_R_loss().backward()

    def optimize_parameters(self):
        # forward
        # Enable autocasting for the forward pass
        with torch.cuda.amp.autocast():
            self.forward()
            self.loss_R = self.compute_R_loss()

        # update R
        self.set_requires_grad(self.netR, True)
        self.optimizer_R.zero_grad()

        # Use scaler to scale the loss before calling backward
        self.scaler.scale(self.loss_R).backward()

        # Use scaler to update the optimizer
        self.scaler.step(self.optimizer_R)

        # Update the scale for next iteration
        self.scaler.update()

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.
        Parameters:
            input (dict): include the data itself and its metadata information.
        A: CityScape image.
        B: Rain streak mask.
        C: Snow mask.
        D: Haze. (Transmission map)
        D1: Light, D2: Medium, D3: Heavy.
        E: Raindrop mask.
        E2: Raindrop texture, paired with E.
        temp_A, used in adding rain streak/snow/haze/raindrop.
        Default normalization makes data between [-1,1], we scale some of them to [0,1] for adding rain components.
        """
        self.temp_A = (input['A'] + 1.0) / 2.0
        self.real_A = input['A'].to(self.device)
        self.real_B = (input['B'] + 1.0) / 2.0 #? why not to device
        self.real_C = (input['C'] + 1.0) / 2.0
        haze = ['D1', 'D2', 'D3']
        if self.isTrain:
            # Randomly choose one.
            self.real_D = (input[choice(haze)] + 1.0) / 2.0
        else:
            # During test, specify a certain intensity.
            self.real_D = (input[haze[self.opt.haze_intensity]] + 1.0) / 2.0
        self.real_E = (input['E1'] + 1.0) / 2.0
        self.real_E2 = (input['E2'] + 1.0) / 2.0
        self.real_H = input['H']
        self.real_I = input['I']
        self.image_paths = input['A_paths']

        input_shape = input['A'].shape
        # Generate Gaussian noise and scale it to the [-1, 1] range
        self.noise = self.generate_gaussian_noise(input_shape)
        self.real_noise = self.scale_to_range(self.noise, -0.5, 0.5)

    def forward(self):
        """
        Run forward pass; called by both functions <optimize_parameters>.
        We have another version of forward (forward_test) used in testing.
        """
        p = torch.rand(self.opt.max_domain)
        # Add rain streak or not. We set rain streak to be always added in the default setting.
        self.label[0] = 1 if p[0] < self.opt.prob1 else 0
        # Add snow/haze/raindrop or not.
        for i in range(1, self.opt.max_domain):
            if p[i] < self.opt.prob2:
                self.label[i] = 1
            else:
                self.label[i] = 0
        label_sum = torch.sum(self.label, 0)
        # Based on the label, starts adding rain components.
        temp = self.temp_A[0].numpy()
        if self.label[0] == 1:
            A = 0.8 + 0.2 * random.random()
            b = self.real_B.numpy()[0]
            temp = self.generate_img(temp, b, A)
        if self.label[1] == 1:
            A = 0.8 + 0.2 * random.random()
            c = self.real_C.numpy()[0]
            temp = self.generate_img(temp, c, A)
        if self.label[2] == 1:
            A = 0.8 + 0.2 * random.random()
            d = self.real_D.numpy()[0]
            temp = self.generate_haze(temp, d, A)
        if self.label[3] == 1:
            e1 = self.real_E.numpy()[0]
            e2 = self.real_E2.numpy()[0]
            e1 = np.transpose(e1, (2, 1, 0))
            e2 = np.transpose(e2, (2, 1, 0))
            position_matrix, alpha = self.get_position_matrix(e2, e1)
            temp = np.transpose(temp, (2, 1, 0))
            temp = self.composition_img(temp, alpha, position_matrix, rate=0.8 + 0.18 * random.random())

        # Convert process image temp to tensor.
        self.real_input = torch.from_numpy(temp.reshape(1, 3, self.width, self.width))
        self.real_B = (self.real_B * 2.0 - 1.0).to(self.device)
        if self.label[0]==0:
            self.real_B = self.real_B * 0.0 - 1.0
        self.real_C = (self.real_C * 2.0 - 1.0).to(self.device)
        if self.label[1]==0:
            self.real_C = self.real_C * 0.0 - 1.0
        self.real_D = (self.real_D * 2.0 - 1.0).to(self.device)
        if self.label[2]==0:
            self.real_D = self.real_D * 0.0 + 1.0
        self.real_E = (self.real_E * 2.0 - 1.0).to(self.device)
        if self.label[3]==0:
            self.real_E = self.real_E * 0.0 - 1.0
        self.real_H = (self.real_H * 2.0 - 1.0).to(self.device)
        self.real_I = (self.real_I * 2.0 - 1.0).to(self.device)
        
        self.real_input = self.real_input.type_as(self.real_A)
        self.real_input = (self.real_input * 2.0 - 1.0).to(self.device)

        self.real_noise = self.real_noise.to(self.device)
        self.input_noised = self.real_input + self.real_noise

        # Clip the pixel values to the [-1, 1] range
        self.input_noised = torch.clamp(self.input_noised, -1, 1).to(self.device)

        # # 计算被改变的像素数量
        # changed_pixels_count = torch.sum(self.cliped_reconstructed_image != self.input_noised).item()
        # # 计算像素总数量
        # total_pixels_count = self.input_noised.numel()
        # # 计算被改变的像素的百分比
        # changed_pixels_percentage = (changed_pixels_count / total_pixels_count) * 100
        # print(f"Changed pixels: {changed_pixels_count} out of {total_pixels_count} total pixels.")
        # print(f"Percentage of pixels changed: {changed_pixels_percentage:.2f}%")

        # stage 0
        prList1 = [torch.FloatTensor([0,0,0,0,0,0,0,0,0,1])
        ]
        prList2 = None
        outList1, _, _ = self.netR(self.input_noised, prList1, prList2)
        self.fake_input = outList1[0]

        # stage 1
        prList1 = [torch.FloatTensor([1,0,0,0,0,0,0,0,0,0]),
                   torch.FloatTensor([0,1,0,0,0,0,0,0,0,0])
        ]
        prList2 = None
        outList1, _, self.pred_label = self.netR(self.fake_input, prList1, prList2)
        self.fake_A, self.fake_B = outList1

        # stage 2
        prList1 = [torch.FloatTensor([0,0,0,0,0,0,0,1,0,0])
        ]
        prList2 = [torch.FloatTensor([0,0,0,0,0,0,0,0,1,0])]
        outList1, outList2, _ = self.netR(self.fake_A, prList1, prList2)
        self.fake_H = outList1[0]
        self.fake_seg = outList2[0]
        self.fake_I = (torch.argmax(self.fake_seg, dim=1).unsqueeze(0)/255.0 * 2.0)-1.0
        self.real_seg = self.clean_seg(self.real_I).to(self.device)
        self.real_I2 = ((self.real_seg).unsqueeze(0)/255.0 * 2.0)-1.0

    def compute_R_loss(self):
        label_sum = torch.sum(self.label, 0)
        self.loss_LnR0 = (self.criterionL1(self.fake_input, self.real_input)+self.criterionL2(self.fake_input, self.real_input)) * self.opt.lambda_Ln
        self.loss_LnR1 = (self.criterionL1(self.fake_A, self.real_A)+self.criterionL2(self.fake_A, self.real_A)) * self.opt.lambda_Ln
        #         + self.criterionL1(self.fake_MixA, self.real_A)+self.criterionL2(self.fake_MixA, self.real_A)
        # self.loss_LnR2 = self.criterionL1(self.fake_input, self.real_input)+self.criterionL2(self.fake_input, self.real_input) \
        #         + self.criterionL1(self.fake_MixI, self.real_input)+self.criterionL2(self.fake_MixI, self.real_input)
        self.loss_LnR2 = (self.criterionL1(self.fake_H, self.real_H)+self.criterionL2(self.fake_H, self.real_H)) * self.opt.lambda_Ln * 0.1
        self.loss_Lnmask = (self.criterionL2(self.fake_B, self.real_B)+self.criterionL2(self.fake_B, self.real_B)) * self.opt.lambda_Ln * self.label[0]
        
        self.loss_VGGR0 = self.criterionVGG(self.fake_input, self.real_input) * self.opt.lambda_VGG
        self.loss_VGGR1 = self.criterionVGG(self.fake_A, self.real_A) * self.opt.lambda_VGG
        # self.loss_VGGR2 = self.criterionVGG(self.fake_input, self.real_input) + self.criterionVGG(self.fake_MixI, self.real_input)
        self.loss_VGGmask = self.criterionVGG(self.fake_B, self.real_B) * self.opt.lambda_VGG * self.label[0]
        
        self.loss_PSNR0 = self.criterionPSNR(self.fake_input, self.real_input)
        self.loss_PSNR1 = self.criterionPSNR(self.fake_A,self.real_A)
        pred_label = self.pred_label.reshape(-1)
        self.loss_BCE = self.criterionBCE(pred_label,self.label) * self.opt.lambda_BCE
        self.loss_CE = self.criterionCE(self.fake_seg, self.real_seg) * self.opt.lambda_BCE
        self.loss_Dice = self.criterionDice(self.fake_seg, self.real_seg) * self.opt.lambda_Dice
        self.loss_Huber = self.criterionHuber(self.fake_H, self.real_H) * self.opt.lambda_Huber

        return (self.loss_LnR0 + self.loss_LnR1 + self.loss_LnR2 + self.loss_Lnmask) / (self.label[0] + 3) \
                + (self.loss_VGGR0 + self.loss_VGGR1 + self.loss_VGGmask) / (self.label[0] + 2) \
                + self.loss_BCE + self.loss_CE + self.loss_Dice + self.loss_Huber

    def clean_seg(self, seg):
        # train_id = [0,16,17,19,20,21,24,26,27,28,29,30,31,32,33,34,36,37,38]
        train_id = [6,7,10,11,12,16,18,19,20,21,22,23,24,25,26,27,30,31,32]
        pic = tensor2im(seg)[:,:,0]
        seg_tensor = torch.from_numpy(pic).unsqueeze(0).clone().type(torch.long)
        
        # Create a mask where seg_tensor values are in train_id
        mask = torch.zeros_like(seg_tensor, dtype=torch.bool)
        for id in train_id:
            mask |= (seg_tensor == id)
        
        # Set all values not within train_id to 0
        seg_tensor[~mask] = 0
        return seg_tensor


    def forward_test(self):
        gt_label = [0] * self.opt.max_domain
        if 'B' in self.opt.test_input:
            gt_label[0] = 1
        if 'C' in self.opt.test_input:
            gt_label[1] = 1
        if 'D' in self.opt.test_input:
            gt_label[2] = 1
        if 'E' in self.opt.test_input:
            gt_label[3] = 1
        temp = self.temp_A[0].numpy()
        if gt_label[0] == 1:
            # Fixed A during testing.
            A = 0.9
            b = self.real_B.numpy()[0]
            temp = self.generate_img(temp, b, A)
        if gt_label[1] == 1:
            # Fixed A during testing.
            A = 0.9
            c = self.real_C.numpy()[0]
            temp = self.generate_img(temp, c, A)
        if gt_label[2] == 1:
            # Fixed A during testing.
            A = 0.9
            d = self.real_D.numpy()[0]
            temp = self.generate_haze(temp, d, A)
        if gt_label[3] == 1:
            e1 = self.real_E.numpy()[0]
            e2 = self.real_E2.numpy()[0]
            e1 = np.transpose(e1, (2, 1, 0))
            e2 = np.transpose(e2, (2, 1, 0))
            position_matrix, alpha = self.get_position_matrix(e2, e1)
            temp = np.transpose(temp, (2, 1, 0))
            # Fixed rate = 0.9 during testing.
            temp = self.composition_img(temp, alpha, position_matrix, rate=0.9)
        self.real_input = torch.from_numpy(temp.reshape(1, 3, self.width, self.width))
        self.real_B = (self.real_B * 2.0 - 1.0).to(self.device)
        self.real_C = (self.real_C * 2.0 - 1.0).to(self.device)
        self.real_D = (self.real_D * 2.0 - 1.0).to(self.device)
        self.real_E = (self.real_E * 2.0 - 1.0).to(self.device)
        self.real_input = self.real_input.type_as(self.real_A)
        self.real_input = (self.real_input * 2.0 - 1.0).to(self.device)
        self.real_H = (self.real_H * 2.0 - 1.0).to(self.device)
        self.real_I = (self.real_I * 2.0 - 1.0).to(self.device)
        input_shape = self.real_input.shape
        self.noise = self.generate_gaussian_noise(input_shape)
        self.real_noise = self.scale_to_range(self.noise, -0.5, 0.5).to(self.device)
        self.input_noised = self.real_input + self.real_noise
        self.input_noised = torch.clamp(self.input_noised, -1, 1).to(self.device)

        # stage 0
        prList1 = [torch.FloatTensor([0,0,0,0,0,0,0,0,0,1])
        ]
        prList2 = None
        outList1, _, _ = self.netR(self.input_noised, prList1, prList2)
        self.fake_input = outList1[0]

        # stage 1
        prList1 = [torch.FloatTensor([1,0,0,0,0,0,0,0,0,0]),
                   torch.FloatTensor([0,1,0,0,0,0,0,0,0,0])
        ]
        prList2 = None
        outList1, _, self.pred_label = self.netR(self.fake_input, prList1, prList2)
        self.fake_A, self.fake_B = outList1

        # stage 2
        prList1 = [torch.FloatTensor([0,0,0,0,0,0,0,1,0,0])
        ]
        prList2 = [torch.FloatTensor([0,0,0,0,0,0,0,0,1,0])]
        outList1, outList2, _ = self.netR(self.fake_A, prList1, prList2)
        self.fake_H = outList1[0]
        self.fake_seg = outList2[0]
        self.fake_I = (torch.argmax(self.fake_seg, dim=1).unsqueeze(0)/255.0 * 2.0)-1.0
        self.real_seg = self.clean_seg(self.real_I).to(self.device)
        self.real_I2 = ((self.real_seg).unsqueeze(0)/255.0 * 2.0)-1.0

        self.criterionCE = torch.nn.CrossEntropyLoss().to(self.device)
        self.criterionDice = DiceLoss(64).to(self.device)
        
        predict_label = torch.where(self.pred_label.squeeze() > 0.5, 1, 0)

        if predict_label.tolist() == gt_label:
            self.accuracy = self.accuracy + 1
        else:
            print('sample:',self.all_count + 1,'; predict_label:',predict_label.tolist(),'; gt_label:',gt_label)
        self.all_count = self.all_count + 1
        
        self.loss_CE = self.criterionCE(self.fake_seg, self.real_seg)
        self.loss_Dice = self.criterionDice(self.fake_seg, self.real_seg)
        self.Seg_CE = self.Seg_CE + self.loss_CE
        self.Seg_Dice = self.Seg_Dice + self.loss_Dice
        
        # Convert the softmax output to class labels
        pred_labels = torch.argmax(self.fake_seg, dim=1)

        # Calculate the metrics
        iou = per_class_iou(pred_labels, self.real_seg)
        pixel_acc = accuracy_multiclass(pred_labels, self.real_seg)

        # Store the metrics
        self.total_iou += iou
        self.total_pixel_acc += pixel_acc

        # print(self.fake_seg.shape, self.real_I.shape, pred_labels.shape)
        # print(self.fake_seg)
        # print(self.real_I)
        # print(self.total_iou, mean_iou, self.total_pixel_acc, mean_pixel_acc)
        
        with open('test_log', 'a') as f:
            if self.all_count == 500:
                print("Accuracy: ", self.accuracy / self.all_count, file=f)
                print("Seg_CE: ", self.Seg_CE / self.all_count, file=f)
                print("mIoU: ", self.total_iou / self.all_count, file=f)
                print("Mean Pixel Acc: ", self.total_pixel_acc / self.all_count, file=f)
                print("Seg_Dice: ", self.Seg_Dice / self.all_count, file=f)
                
                # Print to console as well
                print("Accuracy: ", self.accuracy / self.all_count)
                print("Seg_CE: ", self.Seg_CE / self.all_count)
                print("mIoU: ", self.total_iou / self.all_count)
                print("Mean Pixel Acc: ", self.total_pixel_acc / self.all_count)
                print("Seg_Dice: ", self.Seg_Dice / self.all_count)

    # Rain streak, snow.
    def generate_img(self, img1, img2, A):
        img1 = img1 * (1 - img2) + A * img2
        return img1

    # Haze.
    def generate_haze(self, img1, img2, A):
        img1 = img1 * img2 + A * (1 - img2)
        return img1

    # The following functions are for raindrop, please check more details at ./raindrop.
    def get_position_matrix(self, texture, alpha):
        alpha = cv2.blur(alpha, (5, 5))
        position_matrix = np.mgrid[0:self.width, 0:self.width]
        position_matrix[0, :, :] = position_matrix[0, :, :] + texture[:, :, 2] * (texture[:, :, 0])
        position_matrix[1, :, :] = position_matrix[1, :, :] + texture[:, :, 1] * (texture[:, :, 0])
        position_matrix = position_matrix * (alpha[:, :, 0] > 0.3)

        return position_matrix, alpha

    def composition_img(self, img, alpha, position_matrix, rate, length=2):
        h, w = img.shape[0:2]
        dis_img = img.copy()
        for x in range(h):
            for y in range(w):
                u, v = int(position_matrix[0, x, y] / length), int(position_matrix[1, x, y] / length)
                if u != 0 and v != 0:
                    if (u < h) and (v < w):
                        dis_img[x, y, :] = dis_img[u, v, :]
                    elif u < h:
                        dis_img[x, y, :] = dis_img[u, np.random.randint(0, w - 1), :]
                    elif v < w:
                        dis_img[x, y, :] = dis_img[np.random.randint(0, h - 1), v, :]
        dis_img = cv2.blur(dis_img, (3, 3)) * rate
        img = alpha * dis_img + (1 - alpha) * img
        return np.transpose(img, (2, 1, 0))

    def generate_gaussian_noise(self, shape, mean=0.0, std_dev=0.5):
        """Generate Gaussian noise using PyTorch"""
        mean_tensor = torch.full(shape, float(mean), dtype=torch.float32)
        std_dev_tensor = torch.full(shape, float(std_dev), dtype=torch.float32)
        noise = torch.normal(mean_tensor, std_dev_tensor)
        return noise


    def scale_to_range(self, arr, target_min, target_max):
        """Scale tensor to the target range using PyTorch"""
        scaled_arr = (arr - torch.min(arr)) / (torch.max(arr) - torch.min(arr)) * (target_max - target_min) + target_min
        return scaled_arr

def tensor2im(input_image, imtype=np.uint8):
    """"Converts a Tensor array into a numpy image array.

    Parameters:
        input_image (tensor) --  the input image tensor array
        imtype (type)        --  the desired type of the converted numpy array
    """
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):  # get the data from a variable
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor[0].clamp(-1.0, 1.0).cpu().float().numpy()  # convert it into a numpy array
        if image_numpy.shape[0] == 1:  # grayscale to RGB
            image_numpy = np.tile(image_numpy, (3, 1, 1))
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0  # post-processing: tranpose and scaling
    else:  # if it is a numpy array, do nothing
        image_numpy = input_image
    return image_numpy.astype(imtype)

def per_class_iou(pred_labels, true_labels):
    class_iou = []
    id_list = [6,7,10,11,12,16,18,19,20,21,22,23,24,25,26,27,30,31,32]

    for class_value in range(64):
        if class_value in id_list:
            pred_inds = (pred_labels == class_value)
            true_inds = (true_labels == class_value)
            intersection = (pred_inds & true_inds).sum()
            union = (pred_inds | true_inds).sum()
            if union == 0:
                iou_score = float('nan')  # If there is no ground truth, do not include in evaluation
            else:
                iou_score = float(intersection) / float(max(union, 1))
                class_iou.append(iou_score)
    # print(np.nanmean(class_iou))
    return np.nanmean(class_iou)  # Return the average IoU across all classes

def accuracy_multiclass(pred_labels, true_labels):
    correct_predictions = (pred_labels == true_labels).sum().item()
    total_predictions = true_labels.numel()
    return correct_predictions / total_predictions

class DiceLoss(nn.Module):
    def __init__(self, num_classes):
        super(DiceLoss, self).__init__()
        self.num_classes = num_classes

    def forward(self, input, target):
        target_onehot = F.one_hot(target, self.num_classes).permute(0, 3, 1, 2).float()
        input_soft = F.softmax(input, dim=1)
        
        intersection = input_soft * target_onehot
        intersection = intersection.sum(dim=(2, 3))
        # print(target_onehot.shape, input_soft.shape)
        # print(intersection)
        
        denominator = input_soft + target_onehot
        denominator = denominator.sum(dim=(2, 3))

        dice_score = 2. * intersection / denominator

        dice_score = dice_score.sum(dim=1)/19  # average over classes
        dice_loss = 1. - dice_score.mean(dim=0)  # average over batch
        return dice_loss


    
