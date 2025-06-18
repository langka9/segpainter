import torch
import torch.nn as nn
from .base_model import BaseModel
from . import network, base_function, external_function
from util import task
import itertools
import torch.nn.functional as F
# from criteria import id_loss
from thop import profile



class VSSMSegblkscan(BaseModel):
    """This class implements the VSSM image completion, for 256*256 resolution image inpainting"""
    def name(self):
        return "VSSMSegblkscan"

    @staticmethod
    def modify_options(parser, is_train=True):
        """Add new options and rewrite default values for existing options"""
        if is_train:
            parser.add_argument('--lambda_rec', type=float, default=10.0, help='weight for image reconstruction loss')
            parser.add_argument('--lambda_g', type=float, default=10.0, help='weight for gan loss')
            parser.add_argument('--lambda_gp', type=float, default=10.0, help='weight for wgan-gp loss')
            parser.add_argument('--lambda_per', type=float, default=30.0, help='weight for image perceptual loss')
            parser.add_argument('--lambda_sty', type=float, default=1000.0, help='weight for style loss')

        return parser

    def __init__(self, opt):
        """Initial the pluralistic model"""
        BaseModel.__init__(self, opt)

        self.loss_names = ['app_g', 'ad_g', 'img_d', 'per_g', 'sty_g']
        self.visual_names = ['img_m', 'img_truth', 'img_out']
        self.model_names = ['G', 'D']

        # define the inpainting model
        self.net_G = network.define_g_bs_seg(opt=opt, activation='LeakyReLU', init_type='orthogonal', gpu_ids=opt.gpu_ids)
        # define the discriminator model
        self.net_D = network.define_d(ndf=32, img_f=128, layers=5, model_type='ResDis', init_type='orthogonal', gpu_ids=opt.gpu_ids)

        if self.isTrain:
            # define the loss functions
            self.GANloss = external_function.GANLoss(opt.gan_mode)
            self.L1loss = torch.nn.L1Loss()
            self.L2loss = torch.nn.MSELoss()
            self.resnet_loss = ResNetPL(weights_path='pretrained')
            self.vgg = VGG19()
            # self.id_loss = id_loss.IDLoss().eval()
            if len(self.gpu_ids) > 0:
                self.resnet_loss = self.resnet_loss.cuda()
                self.vgg = self.vgg.cuda()
                # self.id_loss = self.id_loss.cuda()
            # define the optimizer
            self.optimizer_G = torch.optim.Adam(itertools.chain(filter(lambda p: p.requires_grad, self.net_G.parameters())), lr=opt.lr, betas=(0.0, 0.999))
            self.optimizer_D = torch.optim.Adam(itertools.chain(filter(lambda p: p.requires_grad, self.net_D.parameters())), lr=opt.lr, betas=(0.0, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
        # load the pretrained model and schedulers
        else:
            self.GANloss = external_function.GANLoss('wgangp')
        self.setup(opt)

    def set_input(self, input, epoch=0):
        """Unpack input data from the data loader and perform necessary pre-process steps"""
        self.input = input
        self.image_name = self.input['name']
        self.img = input['from_im']
        self.mask = input['mask']
        self.condition = input['condition']

        if len(self.gpu_ids) > 0:
            self.img = self.img.cuda(self.gpu_ids[0])
            self.mask = self.mask.cuda(self.gpu_ids[0])
            self.condition = self.condition.cuda(self.gpu_ids[0])

        # get I_m and I_c for image with mask and complement regions for training
        self.img_truth = self.img
        self.img_m = (1 - self.mask) * self.img_truth if not self.opt.no_use_mask else torch.randn_like(self.img_truth)

    def test(self):
        """Forward function used in test time"""

        flops, params = profile(self.net_G, inputs=(self.img_m, self.condition))
        print('Flops:{} GFlops'.format(flops/1000**3))
        print('Params:{} MB'.format(params/1000**2))
        os._exit(0)

        # save the groundtruth and masked image
        self.save_results(self.img_truth, data_name='truth')
        if not self.opt.no_use_mask:
            self.save_results(self.img_m, data_name='masked')
            self.save_results(self.mask, data_name='mask')

        # encoder process
        img_rec, img_refine = self.net_G(self.img_m, self.condition)
        self.img_out = self.mask * img_refine + (1 - self.mask) * self.img_truth
        self.save_results(self.img_out, data_name='out', masks=self.mask)

    def forward(self):
        """Run forward processing to get the inputs"""
        # encoder process
        img_rec, img_refine = self.net_G(self.img_m, self.condition)
        self.img_coarse = self.mask * img_rec + (1 - self.mask) * self.img_truth
        self.img_out = self.mask * img_refine + (1 - self.mask) * self.img_truth

    def backward_D_basic(self, netD, real, fake):
        """Calculate GAN loss for the discriminator"""
        # Real
        D_real = netD(real)
        D_real_loss = self.GANloss(D_real, True, True)
        # fake
        D_fake = netD(fake.detach())
        D_fake_loss = self.GANloss(D_fake, False, True)
        # loss for discriminator
        D_loss = (D_real_loss + D_fake_loss) * 0.5
        # gradient penalty for wgangp
        if self.opt.gan_mode == 'wgangp':
            gradient_penalty, gradients = external_function.cal_gradient_penalty(netD, real, fake.detach())
            D_loss += gradient_penalty * self.opt.lambda_gp

        D_loss.backward()

        return D_loss

    def backward_D(self):
        """Calculate the GAN loss for the discriminators"""
        base_function._unfreeze(self.net_D)
        loss_img_d1 = self.backward_D_basic(self.net_D, self.img_truth, self.img_coarse)
        loss_img_d2 = self.backward_D_basic(self.net_D, self.img_truth, self.img_out)
        self.loss_img_d = loss_img_d1 + loss_img_d2

    def backward_G(self):
        """Calculate training loss for the generator"""

        # generator adversarial loss
        base_function._freeze(self.net_D)
        # g loss fake
        D_fake = self.net_D(self.img_coarse)
        loss_ad_g1 = self.GANloss(D_fake, True, False) * self.opt.lambda_g
        D_fake = self.net_D(self.img_out)
        loss_ad_g2 = self.GANloss(D_fake, True, False) * self.opt.lambda_g
        self.loss_ad_g = loss_ad_g1 + loss_ad_g2

        # calculate l1 loss
        loss_app_g = self.L1loss(self.img_coarse, self.img_truth) + self.L1loss(self.img_out, self.img_truth)
        self.loss_app_g = loss_app_g * self.opt.lambda_rec

        # calculate perceptual loss
        self.loss_per_g = (self.resnet_loss(self.img_coarse, self.img_truth) + self.resnet_loss(self.img_out, self.img_truth)) * self.opt.lambda_per
        # calculate style loss
        self.loss_sty_g = (self.style_loss(self.img_coarse, self.img_truth, self.vgg) + self.style_loss(self.img_out, self.img_truth, self.vgg)) * self.opt.lambda_sty

        total_loss = 0

        for name in self.loss_names:  # self.loss_names = ['app_g', 'ad_g', 'img_d', 'per_g', 'sty_g', 'id_g']
            if name != 'img_d':
                total_loss += getattr(self, "loss_" + name)

        total_loss.backward()

    def style_loss(self, x_r, x_gt, vgg):
        x_gt_vgg = vgg((x_gt + 1.0) / 2.0)
        x_r_vgg = vgg((x_r + 1.0) / 2.0)
        loss = F.l1_loss(self.compute_gram(x_r_vgg['relu2_2']), self.compute_gram(x_gt_vgg['relu2_2'])) + \
                F.l1_loss(self.compute_gram(x_r_vgg['relu3_4']), self.compute_gram(x_gt_vgg['relu3_4'])) + \
                F.l1_loss(self.compute_gram(x_r_vgg['relu4_4']), self.compute_gram(x_gt_vgg['relu4_4'])) + \
                F.l1_loss(self.compute_gram(x_r_vgg['relu5_2']), self.compute_gram(x_gt_vgg['relu5_2']))

        return loss

    def compute_gram(self, x):
        b, ch, h, w = x.size()
        f = x.view(b, ch, w * h)
        f_T = f.transpose(1, 2)
        G = f.bmm(f_T) / (h * w * ch)

        return G
    
    def optimize_parameters(self):
        """update network weights"""
        # compute the image completion results
        self.forward()
        # optimize the discrinimator network parameters
        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()
        # optimize the completion network parameters
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()


IMAGENET_MEAN = torch.FloatTensor([0.485, 0.456, 0.406])[None, :, None, None]
IMAGENET_STD = torch.FloatTensor([0.229, 0.224, 0.225])[None, :, None, None]
from .ade20k import ModelBuilder

class ResNetPL(nn.Module):
    def __init__(self, weight=1,
                 weights_path=None, arch_encoder='resnet50dilated', segmentation=True):
        super().__init__()
        print('*'*10, weights_path)
        self.impl = ModelBuilder.get_encoder(weights_path=weights_path,
                                             arch_encoder=arch_encoder,
                                             arch_decoder='ppm_deepsup',
                                             fc_dim=2048,
                                             segmentation=segmentation)
        self.impl.eval()
        for w in self.impl.parameters():
            w.requires_grad_(False)

        self.weight = weight

    def forward(self, pred, target, spatial_discounting_mask_tensor=None):
        # pred and target is [0, 1]
        pred = (pred + 1) / 2
        target = (target + 1) / 2
        pred = (pred - IMAGENET_MEAN.to(pred)) / IMAGENET_STD.to(pred)
        target = (target - IMAGENET_MEAN.to(target)) / IMAGENET_STD.to(target)

        pred_feats = self.impl(pred, return_feature_maps=True)
        target_feats = self.impl(target, return_feature_maps=True)

        if spatial_discounting_mask_tensor is not None:
            H = spatial_discounting_mask_tensor.size()[2]
            mse_list = []
            for cur_pred, cur_target in zip(pred_feats, target_feats):
                h = cur_pred.size()[2]
                scale_factor = h / H
                scale_sd_mask_tensor = F.interpolate(spatial_discounting_mask_tensor, scale_factor=scale_factor,
                              mode='bilinear', align_corners=True)
                mse = F.mse_loss(cur_pred * scale_sd_mask_tensor, cur_target * scale_sd_mask_tensor)
                mse_list.append(mse)

            result = torch.stack(mse_list).sum() * self.weight
        else:
            result = torch.stack([F.mse_loss(cur_pred, cur_target) for cur_pred, cur_target in zip(pred_feats, target_feats)]).sum() * self.weight
        return result


import torchvision.models as models
# Assume input range is [0, 1]
class VGG19(torch.nn.Module):
    def __init__(self):
        super(VGG19, self).__init__()
        features = models.vgg19(pretrained=True).features
        self.relu1_1 = torch.nn.Sequential()
        self.relu1_2 = torch.nn.Sequential()

        self.relu2_1 = torch.nn.Sequential()
        self.relu2_2 = torch.nn.Sequential()

        self.relu3_1 = torch.nn.Sequential()
        self.relu3_2 = torch.nn.Sequential()
        self.relu3_3 = torch.nn.Sequential()
        self.relu3_4 = torch.nn.Sequential()

        self.relu4_1 = torch.nn.Sequential()
        self.relu4_2 = torch.nn.Sequential()
        self.relu4_3 = torch.nn.Sequential()
        self.relu4_4 = torch.nn.Sequential()

        self.relu5_1 = torch.nn.Sequential()
        self.relu5_2 = torch.nn.Sequential()
        self.relu5_3 = torch.nn.Sequential()
        self.relu5_4 = torch.nn.Sequential()

        for x in range(2):
            self.relu1_1.add_module(str(x), features[x])

        for x in range(2, 4):
            self.relu1_2.add_module(str(x), features[x])

        for x in range(4, 7):
            self.relu2_1.add_module(str(x), features[x])

        for x in range(7, 9):
            self.relu2_2.add_module(str(x), features[x])

        for x in range(9, 12):
            self.relu3_1.add_module(str(x), features[x])

        for x in range(12, 14):
            self.relu3_2.add_module(str(x), features[x])

        for x in range(14, 16):
            self.relu3_3.add_module(str(x), features[x])

        for x in range(16, 18):
            self.relu3_4.add_module(str(x), features[x])

        for x in range(18, 21):
            self.relu4_1.add_module(str(x), features[x])

        for x in range(21, 23):
            self.relu4_2.add_module(str(x), features[x])

        for x in range(23, 25):
            self.relu4_3.add_module(str(x), features[x])

        for x in range(25, 27):
            self.relu4_4.add_module(str(x), features[x])

        for x in range(27, 30):
            self.relu5_1.add_module(str(x), features[x])

        for x in range(30, 32):
            self.relu5_2.add_module(str(x), features[x])

        for x in range(32, 34):
            self.relu5_3.add_module(str(x), features[x])

        for x in range(34, 36):
            self.relu5_4.add_module(str(x), features[x])

        # don't need the gradients, just want the features
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        relu1_1 = self.relu1_1(x)
        relu1_2 = self.relu1_2(relu1_1)

        relu2_1 = self.relu2_1(relu1_2)
        relu2_2 = self.relu2_2(relu2_1)

        relu3_1 = self.relu3_1(relu2_2)
        relu3_2 = self.relu3_2(relu3_1)
        relu3_3 = self.relu3_3(relu3_2)
        relu3_4 = self.relu3_4(relu3_3)

        relu4_1 = self.relu4_1(relu3_4)
        relu4_2 = self.relu4_2(relu4_1)
        relu4_3 = self.relu4_3(relu4_2)
        relu4_4 = self.relu4_4(relu4_3)

        relu5_1 = self.relu5_1(relu4_4)
        relu5_2 = self.relu5_2(relu5_1)
        relu5_3 = self.relu5_3(relu5_2)
        relu5_4 = self.relu5_4(relu5_3)

        out = {
            'relu1_1': relu1_1,
            'relu1_2': relu1_2,

            'relu2_1': relu2_1,
            'relu2_2': relu2_2,

            'relu3_1': relu3_1,
            'relu3_2': relu3_2,
            'relu3_3': relu3_3,
            'relu3_4': relu3_4,

            'relu4_1': relu4_1,
            'relu4_2': relu4_2,
            'relu4_3': relu4_3,
            'relu4_4': relu4_4,

            'relu5_1': relu5_1,
            'relu5_2': relu5_2,
            'relu5_3': relu5_3,
            'relu5_4': relu5_4,
        }
        return out
