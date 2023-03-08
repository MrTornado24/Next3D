# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""
V4: 
1. 使用分开的latent code控制两个stylegan (不共享梯度);
2. 正交投影的参数从2D改成了3D,使三次投影的变换一致;
3. 三平面变成四平面;
4. 三平面的顺序调换;
"""
from turtle import update
import math
import torch
import numpy as np
import torch.nn.functional as F
from pytorch3d.io import load_obj
import cv2
from torchvision.utils import save_image
from torch_utils import persistence
from training_avatar_texture.networks_stylegan2 import Generator as StyleGAN2Backbone
from training_avatar_texture.volumetric_rendering.renderer import ImportanceRenderer
from training_avatar_texture.volumetric_rendering.ray_sampler import RaySampler
from training_avatar_texture.volumetric_rendering.renderer import Pytorch3dRasterizer, face_vertices, generate_triangles, transform_points, batch_orth_proj, angle2matrix
import dnnlib


@persistence.persistent_class
class TriPlaneGenerator(torch.nn.Module):
    def __init__(self,
        z_dim,                      # Input latent (Z) dimensionality.
        c_dim,                      # Conditioning label (C) dimensionality.
        w_dim,                      # Intermediate latent (W) dimensionality.
        img_resolution,             # Output resolution.
        img_channels,               # Number of output color channels.
        topology_path,              # 
        sr_num_fp16_res     = 0,
        mapping_kwargs      = {},   # Arguments for MappingNetwork.
        rendering_kwargs    = {},
        sr_kwargs = {},
        **synthesis_kwargs,         # Arguments for SynthesisNetwork.
    ):
        super().__init__()
        self.z_dim=z_dim
        self.c_dim=c_dim
        self.w_dim=w_dim
        self.img_resolution=img_resolution
        self.img_channels=img_channels
        self.topology_path = topology_path
        self.renderer = ImportanceRenderer()
        self.ray_sampler = RaySampler()
        self.texture_backbone = StyleGAN2Backbone(z_dim, c_dim, w_dim, img_resolution=256, img_channels=32, mapping_kwargs=mapping_kwargs, **synthesis_kwargs) # render neural texture 
        self.backbone = StyleGAN2Backbone(z_dim, c_dim, w_dim, img_resolution=256, img_channels=32*3, mapping_ws=self.texture_backbone.num_ws*2, mapping_kwargs=mapping_kwargs, **synthesis_kwargs)
        # debug: use splitted w
        self.superresolution = dnnlib.util.construct_class_by_name(class_name=rendering_kwargs['superresolution_module'], channels=32, img_resolution=img_resolution, sr_num_fp16_res=sr_num_fp16_res, sr_antialias=rendering_kwargs['sr_antialias'], **sr_kwargs)
        self.decoder = OSGDecoder(32, {'decoder_lr_mul': rendering_kwargs.get('decoder_lr_mul', 1), 'decoder_output_dim': 32})
        self.neural_rendering_resolution = 64
        self.rendering_kwargs = rendering_kwargs
        self._last_planes = None

        # set pytorch3d rasterizer
        self.uv_resolution = 256
        self.rasterizer = Pytorch3dRasterizer(image_size=256)


        verts, faces, aux = load_obj(self.topology_path)
        uvcoords = aux.verts_uvs[None, ...]      # (N, V, 2)
        uvfaces = faces.textures_idx[None, ...] # (N, F, 3)
        faces = faces.verts_idx[None,...]

        # faces
        dense_triangles = generate_triangles(self.uv_resolution, self.uv_resolution)
        self.register_buffer('dense_faces', torch.from_numpy(dense_triangles).long()[None,:,:].contiguous())
        self.register_buffer('faces', faces)
        self.register_buffer('raw_uvcoords', uvcoords)

        # eye masks
        mask = cv2.imread('D:/projects/DECA/data/uv_face_eye_mask.png').astype(np.float32)/255.; mask = torch.from_numpy(mask[:,:,0])[None,None,:,:].contiguous()
        self.uv_face_mask = F.interpolate(mask, [256, 256])

        # uv coords
        uvcoords = torch.cat([uvcoords, uvcoords[:,:,0:1]*0.+1.], -1) #[bz, ntv, 3]
        uvcoords = uvcoords*2 - 1; uvcoords[...,1] = -uvcoords[...,1]
        face_uvcoords = face_vertices(uvcoords, uvfaces)
        self.register_buffer('uvcoords', uvcoords)
        self.register_buffer('uvfaces', uvfaces)
        self.register_buffer('face_uvcoords', face_uvcoords)

        self.orth_scale = torch.nn.Parameter(torch.tensor([[5.0]]))
        self.orth_shift = torch.nn.Parameter(torch.tensor([[0, -0.01, -0.01]]))

    def mapping(self, z, c, truncation_psi=1, truncation_cutoff=None, update_emas=False):
        if self.rendering_kwargs['c_gen_conditioning_zero']:
            c = torch.zeros_like(c)
        c = c[:, :25]
        return self.backbone.mapping(z, c * self.rendering_kwargs.get('c_scale', 0), truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff, update_emas=update_emas)

    def synthesis(self, ws, c, v, neural_rendering_resolution=None, update_emas=False, cache_backbone=False, use_cached_backbone=False, **synthesis_kwargs):
        batch_size = ws.shape[0]
        eg3d_ws, texture_ws = ws[:, :self.texture_backbone.num_ws], ws[:, self.texture_backbone.num_ws:]
        cam2world_matrix = c[:, :16].view(-1, 4, 4)
        intrinsics = c[:, 16:25].view(-1, 3, 3)

        if neural_rendering_resolution is None:
            neural_rendering_resolution = self.neural_rendering_resolution
        else:
            self.neural_rendering_resolution = neural_rendering_resolution

        # Create a batch of rays for volume rendering
        ray_origins, ray_directions = self.ray_sampler(cam2world_matrix, intrinsics, neural_rendering_resolution)

        # Create triplanes by running StyleGAN backbone
        N, M, _ = ray_origins.shape
        if use_cached_backbone and self._last_planes is not None:
            planes = self._last_planes
            textures = self._last_textures
        else:
            planes = self.backbone.synthesis(eg3d_ws, update_emas=update_emas, **synthesis_kwargs)
            textures = self.texture_backbone.synthesis(texture_ws, update_emas=update_emas, **synthesis_kwargs)
        if cache_backbone:
            self._last_planes = planes
            self._last_texture = textures

        # rasterize to three orthogonal views
        renderings, alphas = self.rasterize_orth(v, textures, batch_size, ws.device)

        # Reshape output into three 32-channel planes
        renderings = renderings.view(len(renderings), 3, 32, renderings.shape[-2], renderings.shape[-1])

        # blend features of neural texture and tri-plane
        planes = planes.view(len(planes), 3, 32, planes.shape[-2], planes.shape[-1])
        blended_planes = renderings * alphas.unsqueeze(2) + planes * (1 - alphas.unsqueeze(2))

        # Perform volume rendering
        feature_samples, depth_samples, weights_samples = self.renderer(blended_planes, self.decoder, ray_origins, ray_directions, self.rendering_kwargs) # channels last

        # Reshape into 'raw' neural-rendered image
        H = W = self.neural_rendering_resolution
        feature_image = feature_samples.permute(0, 2, 1).reshape(N, feature_samples.shape[-1], H, W).contiguous()
        depth_image = depth_samples.permute(0, 2, 1).reshape(N, 1, H, W)

        # Run superresolution to get final image
        rgb_image = feature_image[:, :3]
        sr_image = self.superresolution(rgb_image, feature_image, eg3d_ws, noise_mode=self.rendering_kwargs['superresolution_noise_mode'], **{k:synthesis_kwargs[k] for k in synthesis_kwargs.keys() if k != 'noise_mode'})

        return {'image': sr_image, 'image_raw': rgb_image, 'image_depth': depth_image}

    def rasterize_orth(self, v, textures, batch_size, device):
        # rasterize texture to three orthogonal views
        renderings, alphas = [], []
        tforms = [[0, 0, 180], [90, 0, 180], [0, 90, 180], [0, -90, 180]] # (x, z, y)
        for tform in tforms:
            tform = angle2matrix(torch.tensor(tform).reshape(1, -1)).expand(batch_size, -1, -1).to(device)
            transformed_vertices = (torch.bmm(v, tform) + self.orth_shift.to(device)) * self.orth_scale.to(device)
            transformed_vertices = batch_orth_proj(transformed_vertices, torch.tensor([1., 0, 0]).to(device))
            transformed_vertices[:,:,1:] = -transformed_vertices[:,:,1:]
            transformed_vertices[:,:,2] = transformed_vertices[:,:,2] + 10
            rendering = self.rasterizer(transformed_vertices, self.faces.expand(batch_size, -1, -1), self.face_uvcoords.expand(batch_size, -1, -1, -1), 256, 256)
            alpha_image = rendering[:, -1, :, :][:, None, :, :].detach()
            uvcoords_images = rendering[:, :-1, :, :]; grid = (uvcoords_images).permute(0, 2, 3, 1)[:, :, :, :2]
            mask_face_eye = F.grid_sample(self.uv_face_mask.expand(batch_size,-1,-1,-1).to(device), grid.detach(), align_corners=False) 
            alpha_image = mask_face_eye * alpha_image
            rendering_image = F.grid_sample(textures, grid, align_corners=False)
            renderings.append(rendering_image)
            alphas.append(alpha_image)

        rendering_side = renderings[2] + renderings[3]
        alpha_side = (alphas[2].bool() | alphas[3].bool()).float()
        renderings = torch.cat([renderings[0], renderings[1], rendering_side], 1)
        ### debug: only use dynamic texture in the front view ###
        alphas[1] = torch.zeros_like(alphas[1])
        alpha_side = torch.zeros_like(alpha_side)
        alphas = torch.cat([alphas[0], alphas[1], alpha_side], 1)
        return renderings, alphas

    def sample(self, coordinates, directions, z, c, v, truncation_psi=1, truncation_cutoff=None, update_emas=False, **synthesis_kwargs):
        # Compute RGB features, density for arbitrary 3D coordinates. Mostly used for extracting shapes. 
        ws = self.mapping(z, c, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff, update_emas=update_emas)
        eg3d_ws, texture_ws = ws[:, :self.texture_backbone.num_ws], ws[:, self.texture_backbone.num_ws:]
        planes = self.backbone.synthesis(eg3d_ws, update_emas=update_emas, **synthesis_kwargs)
        textures = self.texture_backbone.synthesis(texture_ws, update_emas=update_emas, **synthesis_kwargs)
        renderings, alphas = self.rasterize_orth(v, textures, batch_size=ws.shape[0], device=ws.device)
        
        planes = planes.view(len(planes), 3, 32, planes.shape[-2], planes.shape[-1])
        renderings = renderings.view(len(renderings), 3, 32, renderings.shape[-2], renderings.shape[-1])
        blended_planes = renderings * alphas.unsqueeze(2) + planes * (1 - alphas.unsqueeze(2))
        return self.renderer.run_model(blended_planes, self.decoder, coordinates, directions, self.rendering_kwargs)

    def sample_mixed(self, coordinates, directions, ws, v, truncation_psi=1, truncation_cutoff=None, update_emas=False, **synthesis_kwargs):
        # Same as sample, but expects latent vectors 'ws' instead of Gaussian noise 'z'
        eg3d_ws, texture_ws = ws[:, :self.texture_backbone.num_ws], ws[:, self.texture_backbone.num_ws:]
        planes = self.backbone.synthesis(eg3d_ws, update_emas = update_emas, **synthesis_kwargs)
        textures = self.texture_backbone.synthesis(texture_ws, update_emas=update_emas, **synthesis_kwargs)
        renderings, alphas = self.rasterize_orth(v, textures, batch_size=ws.shape[0], device=ws.device)
        
        planes = planes.view(len(planes), 3, 32, planes.shape[-2], planes.shape[-1])
        renderings = renderings.view(len(renderings), 3, 32, renderings.shape[-2], renderings.shape[-1])
        blended_planes = renderings * alphas.unsqueeze(2) + planes * (1 - alphas.unsqueeze(2))
        return self.renderer.run_model(blended_planes, self.decoder, coordinates, directions, self.rendering_kwargs)

    def forward(self, z, c, v, truncation_psi=1, truncation_cutoff=None, neural_rendering_resolution=None, update_emas=False, cache_backbone=False, use_cached_backbone=False, **synthesis_kwargs):
        # Render a batch of generated images.
        ws = self.mapping(z, c, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff, update_emas=update_emas)
        return self.synthesis(ws, c, v, update_emas=update_emas, neural_rendering_resolution=neural_rendering_resolution, cache_backbone=cache_backbone, use_cached_backbone=use_cached_backbone, **synthesis_kwargs)


from training.networks_stylegan2 import FullyConnectedLayer

class OSGDecoder(torch.nn.Module):
    def __init__(self, n_features, options):
        super().__init__()
        self.hidden_dim = 64

        self.net = torch.nn.Sequential(
            FullyConnectedLayer(n_features, self.hidden_dim, lr_multiplier=options['decoder_lr_mul']),
            torch.nn.Softplus(),
            FullyConnectedLayer(self.hidden_dim, 1 + options['decoder_output_dim'], lr_multiplier=options['decoder_lr_mul'])
        )
        
    def forward(self, sampled_features, ray_directions, sampled_embeddings=None):
        # Aggregate features
        sampled_features = sampled_features.mean(1)
        x = sampled_features

        N, M, C = x.shape
        x = x.view(N*M, C)

        x = self.net(x)
        x = x.view(N, M, -1)
        rgb = torch.sigmoid(x[..., 1:])*(1 + 2*0.001) - 0.001 # Uses sigmoid clamping from MipNeRF
        sigma = x[..., 0:1]
        return {'rgb': rgb, 'sigma': sigma}
