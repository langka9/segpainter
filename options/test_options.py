from .base_options import BaseOptions
import torch
import re
import os

class TestOptions(BaseOptions):
    def initialize(self,  parser):
        parser = BaseOptions.initialize(self, parser)
        # other options
        parser.add_argument('--dataset_type', default='celeba_encode', type=str, help='Type of dataset/experiment to run')
        parser.add_argument('--con_in_chans', default=19, type=int, help='channels of condition (segmentation map)')
        parser.add_argument('--channel_first', action="store_true", help='if True, [B C H W], otherwise, [B, H, W, C]')

        parser.add_argument('--ntest', type=int, default=float("inf"), help='# of the test examples')
        parser.add_argument('--run_dir', type=str, default='', help='load the latest model from this checkpoint path')
        parser.add_argument('--results_dir', type=str, default='./results/', help='saves results here')
        parser.add_argument('--how_many', type=int, default=50, help='how many test images to run')
        parser.add_argument('--phase', type=str, default='test', help='train, val, test')

        self.isTrain = False

        return parser

    def parse(self):
        """Parse the options"""

        opt = self.gather_options()
        opt.isTrain = self.isTrain

        # set gpu ids
        str_ids = opt.gpu_ids.split(',')
        opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                opt.gpu_ids.append(id)
        if len(opt.gpu_ids):
            torch.cuda.set_device(opt.gpu_ids[0])

        self.opt = opt

        desc = ''
        desc += f'{opt.model}'
        desc += f'-{opt.dataset_type}'
        desc += f'-{opt.fineSize}'
        if opt.no_use_mask:
            desc += '-noise'
        
        self.opt.name = desc

        # Pick output directory.
        assert self.opt.run_dir != ''

        self.print_options(opt)

        return self.opt

