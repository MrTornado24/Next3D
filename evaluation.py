from curses import meta
import cv2
import click
import torch
import os
import copy
import legacy
import glob
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms, utils
import dnnlib
import json
import imageio
from torch_utils import misc
from camera_utils import LookAtPoseSampler, FOV_to_intrinsics
from gen_videos import parse_range, parse_tuple
from training_next3d.triplane_v13_neural_blending_shallow import TriPlaneGenerator
from torchvision.utils import save_image
import sklearn.metrics
import math 


@click.command()
@click.option("--vert_root", type=str, default=None)
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
@click.option('--num_samples', 'num_samples', help='number of samples', required=True, type=int, metavar='INT')
@click.option('--outdir', help='Output directory', type=str, required=True, metavar='DIR')
@click.option('--reload_modules', help='Overload persistent modules?', type=bool, required=False, metavar='BOOL', default=False, show_default=True)

def run_evaluation(vert_root, network_pkl, num_samples, outdir, reload_modules):
    device = torch.device('cuda')

    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device)
    
    if reload_modules:
        print("Reloading Modules!")
        G_new = TriPlaneGenerator(*G.init_args, **G.init_kwargs).eval().requires_grad_(False).to(device)
        misc.copy_params_and_buffers(G, G_new, require_all=True)
        G_new.neural_rendering_resolution = G.neural_rendering_resolution
        G_new.rendering_kwargs = G.rendering_kwargs
        G = G_new

    os.makedirs(outdir, exist_ok=True)

    with open('data/ffhq/images512x512/dataset.json', 'rb') as f:
        label_list = json.load(f)['labels']
    with open('data/ffhq/images512x512/dataset_exp_eye.json', 'rb') as f:
        exp_list = json.load(f)['labels']
    exp_list = dict(exp_list)
    vert_list = [os.path.relpath(os.path.join(root, fname), start=vert_root) for root, _dirs, files in os.walk(vert_root) for fname in files]

    sample_list = {'pose': [], 'exp': []}
    for i in range(num_samples):
        vert_path = vert_list[np.random.randint(len(vert_list))]
        v = []
        with open(os.path.join(vert_root, vert_path), "r") as f:
            while True:
                line = f.readline()
                if line == "":
                    break
                if line[:2] == "v ":
                    v.append([float(x) for x in line.split()[1:]])
        v = np.array(v).reshape((-1, 3))
        v = torch.from_numpy(v).cuda().float().unsqueeze(0)

        lms_root = vert_root.replace('mesheseye', 'lms')
        lms = np.loadtxt(os.path.join(lms_root, vert_path.replace('.obj', '.txt')))
        lms = torch.from_numpy(lms).cuda().float().unsqueeze(0)
        v = torch.cat((v, lms), 1)

        c = torch.tensor(label_list[np.random.randint(len(label_list))][1]).unsqueeze(0).cuda()
        z = torch.randn([1, G.z_dim]).cuda()
        img = G(z, c, v, noise_mode='const')['image']
        save_image(img, f'{outdir}/{i:08d}.png', normalize=True, range=(-1, 1))
        exp = exp_list[vert_path.replace('obj', 'png').replace('\\', '/')]
        sample_list['pose'].append(c.cpu().detach().numpy()[0].tolist())
        sample_list['exp'].append(exp)

    
    with open(f'evaluation_{num_samples}.json', "w") as f:
        json.dump(sample_list, f, indent=4)


@click.command()
@click.option("--real_data", type=str, default=None)
@click.option('--fake_data', type=str, default=None)

def cal_evaluation(real_data, fake_data):
    with open(fake_data, 'rb') as f:
        fakes = json.load(f)
    fake_exps = fakes['exp']
    fake_poses = fakes['pose']
    with open(real_data, 'rb') as f:
        reals = json.load(f)
    real_exps = reals['exps']
    real_poses = reals['poses']

    # AED = torch.sum(torch.sum((fake_exps - real_exps)**2, dim=-1)) / (fake_exps.shape[0])
    # APD = torch.sum(torch.sum((fake_poses - real_poses)**2, dim=-1)) / (fake_poses.shape[0])
    
    AED = math.sqrt(sklearn.metrics.mean_squared_error(np.array(real_exps), np.array(fake_exps)[:, :50]))

    APD = math.sqrt(sklearn.metrics.mean_squared_error(np.array(real_poses)[:, :3], np.array(fake_exps)[:, 50:53]))

    print(f"AED: {AED}; APD: {APD}")
    



    

if __name__ == "__main__":
    # run_evaluation()
    cal_evaluation()