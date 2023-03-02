'''
Copyright (C) 2010-2021 Alibaba Group Holding Limited.
'''

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from mmdet.models.builder import BACKBONES
from torch import nn

from . import PlainNet
from .PlainNet import (_create_netblock_list_from_str_, basic_blocks,
                      parse_cmd_options, super_blocks)

sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def parse_cmd_options(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--no_BN', action='store_true')
    parser.add_argument('--no_reslink', action='store_true')
    parser.add_argument('--use_se', action='store_true')
    module_opt, _ = parser.parse_known_args(argv)
    return module_opt


@BACKBONES.register_module()
class MasterNet(PlainNet.PlainNet):
    def __init__(self,
                 argv=None,
                 opt=None,
                 num_classes=None,
                 plainnet_struct=None,
                 no_create=False,
                 no_reslink=None,
                 no_BN=None,
                 use_se=None):

        if argv is not None:
            module_opt = parse_cmd_options(argv)
        else:
            module_opt = None

        if no_BN is None:
            if module_opt is not None:
                no_BN = module_opt.no_BN
            else:
                no_BN = False

        if no_reslink is None:
            if module_opt is not None:
                no_reslink = module_opt.no_reslink
            else:
                no_reslink = False

        if use_se is None:
            if module_opt is not None:
                use_se = module_opt.use_se
            else:
                use_se = False
        plainnet_struct = plainnet_struct

        self.num_classes = 2048
        self.last_channels = 2048
        super(MasterNet, self).__init__(argv=argv,
                                        opt=opt,
                                        num_classes=num_classes,
                                        plainnet_struct=plainnet_struct,
                                        no_create=no_create,
                                        no_reslink=no_reslink,
                                        no_BN=no_BN,
                                        use_se=use_se)

        self.no_create = no_create
        self.no_reslink = no_reslink
        self.no_BN = no_BN
        self.use_se = use_se

        block_cfg = None
        if block_cfg is None:
            stage_planes = [
                16, 40, 64, 96, 224, 2048
            ]  #[16, 40, 64, 96, 224, 2048]  #0.25 default  #[16, 16, 40, 72, 152, 288]
            stage_blocks = [1, 5, 5, 1]
        else:
            stage_planes = block_cfg['stage_planes']
            stage_blocks = block_cfg['stage_blocks']

        # bn eps
        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eps = 1e-3

        #self.stage_layers = extract_stage_features_and_logit()

    def extract_stage_features_and_logit(self, x, target_downsample_ratio=4):
        stage_features_list = []
        image_size = x.shape[2]
        output = x
        block_id = 0
        for block_id, the_block in enumerate(self.block_list):
            output = the_block(output)
            #import pdb; pdb.set_trace()
            dowsample_ratio = round(image_size / output.shape[2])
            if dowsample_ratio == target_downsample_ratio:
                stage_features_list.append(output)
                target_downsample_ratio *= 2
            pass
        pass

        #import pdb; pdb.set_trace()
        #output = F.adaptive_avg_pool2d(output, output_size=1)
        #output = torch.flatten(output, 1)
        #logit = self.fc_linear(output)
        #print("stage_features_list:", stage_features_list)
        return stage_features_list

    def forward(self, x):
        output = self.extract_stage_features_and_logit(x)

        #output = x
        #for block_id, the_block in enumerate(self.block_list):
        #import pdb; pdb.set_trace()
        #    output = the_block(output)

        #output = F.adaptive_avg_pool2d(output, output_size=1)

        #output = torch.flatten(output, 1)
        #output = self.fc_linear(output)
        return tuple(output)

    def forward_pre_GAP(self, x):
        output = x
        for the_block in self.block_list:
            output = the_block(output)
        return output

    def get_FLOPs(self, input_resolution):
        the_res = input_resolution
        the_flops = 0
        for the_block in self.block_list:
            the_flops += the_block.get_FLOPs(the_res)
            the_res = the_block.get_output_resolution(the_res)

        return the_flops

    def get_model_size(self):
        the_size = 0
        for the_block in self.block_list:
            the_size += the_block.get_model_size()

        return the_size

    def get_num_layers(self):
        num_layers = 0
        for block in self.block_list:
            assert isinstance(block, super_blocks.PlainNetSuperBlockClass)
            num_layers += block.sub_layers
        return num_layers

    def replace_block(self, block_id, new_block):
        self.block_list[block_id] = new_block

        if block_id < len(self.block_list) - 1:
            if self.block_list[block_id +
                               1].in_channels != new_block.out_channels:
                self.block_list[block_id + 1].set_in_channels(
                    new_block.out_channels)
        else:
            assert block_id == len(self.block_list) - 1
            self.last_channels = self.block_list[-1].out_channels

        self.module_list = nn.ModuleList(self.block_list)

    def split(self, split_layer_threshold):
        new_str = ''
        for block in self.block_list:
            new_str += block.split(split_layer_threshold=split_layer_threshold)
        return new_str

    def init_weights(self, pretrained=None):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight.data, gain=3.26033)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(
                    m.weight, 0, 3.26033 *
                    np.sqrt(2 / (m.weight.shape[0] + m.weight.shape[1])))
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.zeros_(m.bias)
            else:
                pass

        for superblock in self.block_list:
            if not isinstance(superblock,
                              super_blocks.PlainNetSuperBlockClass):
                continue
            for block in superblock.block_list:
                if not (isinstance(block, basic_blocks.ResBlock)
                        or isinstance(block, basic_blocks.ResBlockProj)):
                    continue
                # print('---debug set bn weight zero in resblock {}:{}'.format(superblock, block))
                last_bn_block = None
                for inner_resblock in block.block_list:
                    if isinstance(inner_resblock, basic_blocks.BN):
                        last_bn_block = inner_resblock
                    pass
                pass  # end for
                assert last_bn_block is not None
                # print('-------- last_bn_block={}'.format(last_bn_block))
                nn.init.zeros_(last_bn_block.netblock.weight)
