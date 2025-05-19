
import torch
import torch.nn as nn

from basicsr.archs import Segs
from basicsr.metrics.stream_metrics import StreamSegMetrics
from torch.nn.parallel import DataParallel, DistributedDataParallel



class SegFeatureExtractor(nn.Module):

    def __init__(self, cfg):
        super(SegFeatureExtractor, self).__init__()

        self.cfg = cfg
        model_map = {
            'deeplabv3_resnet50': Segs.deeplabv3_resnet50,
            'deeplabv3plus_resnet50': Segs.deeplabv3plus_resnet50,
            'deeplabv3_resnet101': Segs.deeplabv3_resnet101,
            'deeplabv3plus_resnet101': Segs.deeplabv3plus_resnet101,
            'deeplabv3_mobilenet': Segs.deeplabv3_mobilenet,
            'deeplabv3plus_mobilenet': Segs.deeplabv3plus_mobilenet
        }
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.segnet = model_map[cfg["name"]](num_classes=cfg["num_classes"], output_stride=cfg["output_stride"])
        # self.segnet.to(self.device)
        self.segnet = self.model_to_device(self.segnet)

        # Set up metrics
        self.metrics_seg = StreamSegMetrics(cfg["num_classes"])

        # restore
        checkpoint = torch.load(cfg["ckpt"], map_location=torch.device('cpu'))
        self.segnet.load_state_dict(checkpoint["model_state"])
        print("Segmentation Model restored from %s" % cfg["ckpt"])

        self.segnet.eval()
        for p in self.segnet.parameters():
            p.requires_grad = False

        del checkpoint  # free memory

    def forward(self, x):

        # output = {}

        out = self.segnet.backbone(x)

        # output['out'] = out

        return out

    def model_to_device(self, net):
        """Model to device. It also warps models with DistributedDataParallel
        or DataParallel.

        Args:
            net (nn.Module)
        """
        net = net.to(self.device)
        # if self.cfg['dist']:
        #     find_unused_parameters = self.cfg.get('find_unused_parameters', False)
        #     net = DistributedDataParallel(
        #         net, device_ids=[torch.cuda.current_device()], find_unused_parameters=find_unused_parameters)
        # elif self.cfg['num_gpu'] > 1:
        #     net = DataParallel(net, device_ids=[0,1])
        return net
