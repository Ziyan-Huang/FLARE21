from nnunet.training.network_training.nnUNetTrainerV2 import nnUNetTrainerV2
import torch
from nnunet.network_architecture.LetsGo_UNet import LetsGo_UNet
from nnunet.utilities.nd_softmax import softmax_helper

class LetsGoTrainer(nnUNetTrainerV2):
    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False):

        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data,
                         deterministic, fp16)
        self.arch_list = [[48, 48], [96, 96], [192, 192, 192], [384, 384, 384], [384, 384, 384], [192, 320, 128], [384, 320, 320], [384, 384, 384], [192, 192, 192], [96], [48]] 

    def initialize_network(self):

        self.network = LetsGo_UNet(self.num_input_channels, self.num_classes,
                                    len(self.net_num_pool_op_kernel_sizes),
                                    self.arch_list,
                                    self.net_num_pool_op_kernel_sizes, 
                                    self.net_conv_kernel_sizes)
        print('current arch:', self.arch_list)
        if torch.cuda.is_available():
            self.network.cuda()
        self.network.inference_apply_nonlin = softmax_helper