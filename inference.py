import os
import imageio
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pprint import pprint
import torch
from easydict import EasyDict
from torchvision import transforms
import traceback
from easydict import EasyDict as edict
try:
    from transforms import sep_transforms
    from utils.flow_utils import flow_to_image, resize_flow
    from utils.torch_utils import restore_model
    from models.pwclite import PWCLite
except:
    traceback.print_exc()
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
    '''
    python3 inference.py 
    -m checkpoints/KITTI15/pwclite_ar.tar 
    -s 384 640
    -i examples/img1.png examples/img2.png
    '''

    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', default='checkpoints/KITTI15/pwclite_ar.tar')
    parser.add_argument('-s', '--test_shape', default=[384, 640], type=int, nargs=2)
    parser.add_argument('-i', '--img_list', nargs='+',
                        default=['examples/img1.png', 'examples/img2.png'])
    args = parser.parse_args()
    # print (args)
    # assert False

    cfg = {
        'model': {
            'upsample': True,
            'n_frames': 2, #len(args.img_list),
            'reduce_dense': True,
            'device': None
        },
        'pretrained_model': 'checkpoints/KITTI15/pwclite_ar.tar', # args.model,
        'test_shape': [384, 640] # args.test_shape,
    }

    device = torch.device("cuda")

    ts = TestHelper(cfg, device)

    #----------
    # img_list = ['examples/img1.png', 'examples/img2.png']
    # imgs = [imageio.imread(img).astype(np.float32) for img in img_list]

    # img_curr = ts.input_transform(imgs[0]).unsqueeze(0)
    # print (img_curr.shape)

    # imgs = [ts.input_transform(img).unsqueeze(0) for img in imgs]
    # img_pair = torch.cat(imgs, 1).to(device)
    # print (img_pair.shape)
    # assert False
    #----------

    dir_images = '/home/ubuntu/dataset/freeweights/Concentration_Curl/gxYRRLluNWg/0.000_10.000/frames'
    img_list = os.listdir(dir_images)
    img_list = [dir_images + '/'+item for item in img_list if item[-4:] == '.png']
    img_list.sort()
    # pprint (img_list)

    if 0:
        dir_result = './demo_ref'
        os.makedirs(dir_result, exist_ok=True)
        dir_vis = dir_result +'/viz'
        os.makedirs(dir_vis, exist_ok=True)

        fig = plt.figure()
        for i in range(0, len(img_list)-1):
            imgs = [imageio.imread(img_list[i]).astype(np.float32),
                    imageio.imread(img_list[i+1]).astype(np.float32)]
            h, w = imgs[0].shape[:2]

            flow_12 = ts.run(imgs)['flows_fw'][0]

            flow_12 = resize_flow(flow_12, (h, w))

            np_flow_12 = flow_12[0].detach().cpu().numpy().transpose([1, 2, 0])
            file_of = f'{dir_result}/result_{i}.npy'
            np.save(file_of, np_flow_12)
            print (f'save in ... {dir_result}/result_{i}.npy')

            vis_flow = flow_to_image(np_flow_12)
            plt.imsave(f'{dir_vis}/result_{i}.png', vis_flow)
            print (f'save in ... {dir_vis}/result_{i}.png')

        assert False

    num_batch = 15
    dir_result = f'./demo_b_{num_batch}'
    os.makedirs(dir_result, exist_ok=True)
    dir_vis = dir_result +'/viz'
    os.makedirs(dir_vis, exist_ok=True)

    h, w = imageio.imread(img_list[0]).shape[:2]
    print (h,w)

    fig = plt.figure()

    for i in range(0, len(img_list)-1, num_batch):
        img_curr = [img_list[j] for j in range(i, i+num_batch) if j < len(img_list)-1]
        pprint (img_curr)
        img_curr = [imageio.imread(img).astype(np.float32) for img in img_curr]

        img_next = [img_list[j+1] for j in range(i, i+num_batch) if j < len(img_list)-1]
        pprint (img_next)
        img_next = [imageio.imread(img).astype(np.float32) for img in img_next]
        # assert False

        img_curr = [ts.input_transform(img).unsqueeze(0) for img in img_curr]
        img_curr = torch.cat(img_curr, 0).to(device)
        
        img_next = [ts.input_transform(img).unsqueeze(0) for img in img_next]
        img_next = torch.cat(img_next, 0).to(device)

        img_pair = [img_curr, img_next]
        img_pair = torch.cat(img_pair, 1).to(device)
        # print (img_curr.shape)
        # print (img_next.shape)
        # print (img_pair.shape)

        flow_12 = ts.model(img_pair)['flows_fw'][0]
        # for k in range(len(flow_12)):
        #     print (flow_12[k].shape)
        # print (flow_12.shape)

        flow_12 = resize_flow(flow_12, (h, w))
        # print (flow_12.shape)

        # np_flow_12 = flow_12[0].detach().cpu().numpy().transpose([1, 2, 0])
        np_flow_12 = flow_12.detach().cpu().numpy().transpose([0, 2, 3, 1])
        # print (np_flow_12.shape)

        for k in range(np_flow_12.shape[0]):
            file_of = f'{dir_result}/result_{i+k}.npy'
            np.save(file_of, np_flow_12[k])
            print (f'save in ... {dir_result}/result_{i+k}.npy')

            vis_flow = flow_to_image(np_flow_12[k])
            plt.imsave(f'{dir_vis}/result_{i+k}.png', vis_flow)
            print (f'save in ... {dir_vis}/result_{i+k}.png')
        print ('--------------')
        # assert False