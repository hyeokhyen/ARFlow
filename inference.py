import imageio
import argparse
import numpy as np
import matplotlib.pyplot as plt

import torch
from easydict import EasyDict
from torchvision import transforms
try:
    from transforms import sep_transforms
    from utils.flow_utils import flow_to_image, resize_flow
    from utils.torch_utils import restore_model
    from models.pwclite import PWCLite
except:
    from .transforms import sep_transforms
    from .utils.flow_utils import flow_to_image, resize_flow
    from .utils.torch_utils import restore_model
    from .models.pwclite import PWCLite


class TestHelper():
    def __init__(self, cfg, device):
        self.cfg = EasyDict(cfg)
        self.cfg.model.device = device
        self.device = device
        # self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")        
        self.model = self.init_model()
        self.input_transform = transforms.Compose([
            sep_transforms.Zoom(*self.cfg.test_shape),
            sep_transforms.ArrayToTensor(),
            transforms.Normalize(mean=[0, 0, 0], std=[255, 255, 255]),
        ])

    def init_model(self):
        model = PWCLite(self.cfg.model)
        # print('Number fo parameters: {}'.format(model.num_parameters()))
        model = restore_model(model, self.cfg.pretrained_model, self.device)
        model = model.to(self.device)
        model.eval()
        return model

    def run(self, imgs):
        imgs = [self.input_transform(img).unsqueeze(0) for img in imgs]
        img_pair = torch.cat(imgs, 1).to(self.device)
        return self.model(img_pair)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', default='checkpoints/KITTI15/pwclite_ar.tar')
    parser.add_argument('-s', '--test_shape', default=[384, 640], type=int, nargs=2)
    parser.add_argument('-i', '--img_list', nargs='+',
                        default=['examples/img1.png', 'examples/img2.png'])
    args = parser.parse_args()

    cfg = {
        'model': {
            'upsample': True,
            'n_frames': len(args.img_list),
            'reduce_dense': True,
            'device': None
        },
        'pretrained_model': args.model,
        'test_shape': args.test_shape,
    }

    device = torch.device("cpu")

    ts = TestHelper(cfg, device)

    imgs = [imageio.imread(img).astype(np.float32) for img in args.img_list]
    h, w = imgs[0].shape[:2]
    print (h,w)
    print (imgs[0].shape)
    # print (imgs)
    assert False

    flow_12 = ts.run(imgs)['flows_fw'][0]

    flow_12 = resize_flow(flow_12, (h, w))
    np_flow_12 = flow_12[0].detach().cpu().numpy().transpose([1, 2, 0])

    vis_flow = flow_to_image(np_flow_12)

    fig = plt.figure()
    plt.imsave('examples/result.png', vis_flow)
    print ('save in ... examples/result.png')
    # plt.imshow(vis_flow)
    # plt.show()
