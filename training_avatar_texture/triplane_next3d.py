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
1. 使用相同的latent code控制两个stylegan (不共享梯度);
2. 正交投影的参数从2D改成了3D,使三次投影的变换一致;
3. 三平面变成四平面;
4. 三平面的顺序调换;
5. 生成嘴部的动态纹理, 和静态纹理融合 (Styleunet)
"""
from os import device_encoding
from turtle import update
import math
import torch
import numpy as np
import torch.nn.functional as F
from pytorch3d.io import load_obj
import cv2
from torchvision.utils import save_image

import dnnlib
from torch_utils import persistence
from training_avatar_texture.networks_stylegan2 import Generator as StyleGAN2Backbone
from training_avatar_texture.networks_stylegan2_styleunet import Generator as CondStyleGAN2Backbone
from training_avatar_texture.volumetric_rendering.renderer import ImportanceRenderer
from training_avatar_texture.volumetric_rendering.ray_sampler import RaySampler
from training_avatar_texture.volumetric_rendering.renderer import Pytorch3dRasterizer, face_vertices, generate_triangles, transform_points, batch_orth_proj, angle2matrix
from training_avatar_texture.volumetric_rendering.renderer import fill_mouth


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
        self.mouth_backbone = CondStyleGAN2Backbone(z_dim, c_dim, w_dim, img_resolution=256, img_channels=32, in_size=64, final_size=4, cond_channels=32, num_cond_res=64, mapping_kwargs=mapping_kwargs, **synthesis_kwargs)
        self.backbone = StyleGAN2Backbone(z_dim, c_dim, w_dim, img_resolution=256, img_channels=32*3, mapping_ws=self.texture_backbone.num_ws*2, mapping_kwargs=mapping_kwargs, **synthesis_kwargs)
        # debug: use splitted w
        self.superresolution = dnnlib.util.construct_class_by_name(class_name=rendering_kwargs['superresolution_module'], channels=32, img_resolution=img_resolution, sr_num_fp16_res=sr_num_fp16_res, sr_antialias=rendering_kwargs['sr_antialias'], **sr_kwargs)
        self.decoder = OSGDecoder(32, {'decoder_lr_mul': rendering_kwargs.get('decoder_lr_mul', 1), 'decoder_output_dim': 32})
        self.neural_rendering_resolution = 64
        self.rendering_kwargs = rendering_kwargs
        self._last_planes = None
        self.load_lms = True

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
        mask = cv2.imread('data/ffhq/uv_face_eye_mask.png').astype(np.float32)/255.; mask = torch.from_numpy(mask[:,:,0])[None,None,:,:].contiguous()
        self.uv_face_mask = F.interpolate(mask, [256, 256])

        # mouth mask
        self.fill_mouth = True

        # uv coords
        uvcoords = torch.cat([uvcoords, uvcoords[:,:,0:1]*0.+1.], -1) #[bz, ntv, 3]
        uvcoords = uvcoords*2 - 1; uvcoords[...,1] = -uvcoords[...,1]
        face_uvcoords = face_vertices(uvcoords, uvfaces)
        self.register_buffer('uvcoords', uvcoords)
        self.register_buffer('uvfaces', uvfaces)
        self.register_buffer('face_uvcoords', face_uvcoords)

        self.orth_scale = torch.tensor([[5.0]])
        self.orth_shift = torch.tensor([[0, -0.01, -0.01]])

        # neural blending
        self.neural_blending = CondStyleGAN2Backbone(z_dim, c_dim, w_dim, cond_channels=32, img_resolution=256, img_channels=32, in_size=256, final_size=32, num_cond_res=256, mapping_kwargs=mapping_kwargs, **synthesis_kwargs)

    def mapping(self, z, c, truncation_psi=1, truncation_cutoff=None, update_emas=False):
        if self.rendering_kwargs['c_gen_conditioning_zero']:
            c = torch.zeros_like(c)
        c = c[:, :25] # remove expression labels
        return self.backbone.mapping(z, c * self.rendering_kwargs.get('c_scale', 0), truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff, update_emas=update_emas)

    def synthesis(self, ws, c, v, neural_rendering_resolution=None, update_emas=False, cache_backbone=False, use_cached_backbone=False, **synthesis_kwargs):
        # split vertices and landmarks 
        if self.load_lms:
            v, lms = v[:, :5023], v[:, 5023:]
        
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
        textures = self.texture_backbone.synthesis(texture_ws, update_emas=update_emas, **synthesis_kwargs)

        # rasterize to three orthogonal views
        rendering_views = [
            [0, 0, 0],
            [0, 90, 0],
            [0, -90, 0],
            [90, 0, 0]
        ]
        rendering_images, alpha_images, uvcoords_images, lm2ds = self.rasterize(v, lms, textures, rendering_views, batch_size, ws.device)
        
        # generate front mouth masks
        rendering_image_front = rendering_images[0]
        mouths_mask = self.gen_mouth_mask(lm2ds[0])
        rendering_mouth = [rendering_image_front[i:i+1, :][:, :, m[0]:m[1], m[2]:m[3]] for i, m in enumerate(mouths_mask)]
        rendering_mouth = torch.cat([torch.nn.functional.interpolate(uv, size=(64, 64), mode='bilinear', antialias=True) for uv in rendering_mouth], 0)
        
        # generate mouth front plane and integrate back to face front plane
        mouths_plane = self.mouth_backbone.synthesis(rendering_mouth, eg3d_ws, update_emas=update_emas, **synthesis_kwargs)
        rendering_stitch = []
        for rendering, m, mouth_plane in zip(rendering_image_front, mouths_mask, mouths_plane):
            rendering = rendering.unsqueeze(0)
            dummy = torch.zeros_like(rendering)
            dummy[:, :] = rendering
            dummy[:, :, m[0]:m[1], m[2]:m[3]] = torch.nn.functional.interpolate(mouth_plane.unsqueeze(0), size=(m[1]-m[0], m[1]-m[0]), mode='bilinear', antialias=True)
            rendering_stitch.append(dummy)
        rendering_stitch = torch.cat(rendering_stitch, 0)
        rendering_stitch = self.neural_blending.synthesis(rendering_stitch, eg3d_ws, update_emas=update_emas, **synthesis_kwargs)

        # generate static triplane
        static_plane = self.backbone.synthesis(eg3d_ws, update_emas=update_emas, **synthesis_kwargs)
        static_plane = static_plane.view(len(static_plane), 3, 32, static_plane.shape[-2], static_plane.shape[-1])
        
        # blend features of neural texture and tri-plane
        alpha_image = torch.cat(alpha_images, 1).unsqueeze(2)
        rendering_stitch = torch.cat((rendering_stitch, rendering_images[1], rendering_images[2]), 1)
        rendering_stitch = rendering_stitch.view(*static_plane.shape)
        blended_planes = rendering_stitch * alpha_image + static_plane * (1 - alpha_image)

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

    def rasterize(self, v, lms, textures, tforms, batch_size, device):
        rendering_images, alpha_images, uvcoords_images, transformed_lms = [], [], [], []

        for tform in tforms:
            v_flip, lms_flip = v.detach().clone(), lms.detach().clone()
            v_flip[..., 1] *= -1; lms_flip[..., 1] *= -1
            # rasterize texture to three orthogonal views
            tform = angle2matrix(torch.tensor(tform).reshape(1, -1)).expand(batch_size, -1, -1).to(device)
            transformed_vertices = (torch.bmm(v_flip, tform) + self.orth_shift.to(device)) * self.orth_scale.to(device)
            transformed_vertices = batch_orth_proj(transformed_vertices, torch.tensor([1., 0, 0]).to(device))
            transformed_vertices[:,:,1:] = -transformed_vertices[:,:,1:]
            transformed_vertices[:,:,2] = transformed_vertices[:,:,2] + 10

            transformed_lm = (torch.bmm(lms_flip, tform) + self.orth_shift.to(device)) * self.orth_scale.to(device)
            transformed_lm = batch_orth_proj(transformed_lm, torch.tensor([1., 0, 0]).to(device))[:, :, :2]
            transformed_lm[:,:,1:] = -transformed_lm[:,:,1:]

            faces = self.faces.detach().clone()[..., [0,2,1]].expand(batch_size, -1, -1)
            attributes = self.face_uvcoords.detach().clone()[:, :, [0,2,1]].expand(batch_size, -1, -1, -1)

            rendering = self.rasterizer(transformed_vertices, faces, attributes, 256, 256)
            alpha_image = rendering[:, -1, :, :][:, None, :, :].detach()
            uvcoords_image = rendering[:, :-1, :, :]; grid = (uvcoords_image).permute(0, 2, 3, 1)[:, :, :, :2]
            mask_face_eye = F.grid_sample(self.uv_face_mask.expand(batch_size,-1,-1,-1).to(device), grid.detach(), align_corners=False) 
            alpha_image = mask_face_eye * alpha_image
            if self.fill_mouth:
                alpha_image = fill_mouth(alpha_image)
            uvcoords_image = mask_face_eye * uvcoords_image
            rendering_image = F.grid_sample(textures, grid.detach(), align_corners=False)

            rendering_images.append(rendering_image)
            alpha_images.append(alpha_image)
            uvcoords_images.append(uvcoords_image)
            transformed_lms.append(transformed_lm)

        rendering_image_side = rendering_images[1] + rendering_images[2] # concatenate two side-view renderings
        alpha_image_side = (alpha_images[1].bool() | alpha_images[1].bool()).float()
        rendering_images = [rendering_images[0], rendering_image_side, rendering_images[3]]
        alpha_images = [alpha_images[0], alpha_image_side, alpha_images[3]]

        return rendering_images, alpha_images, uvcoords_images, transformed_lms

    def sample(self, coordinates, directions, z, c, v, truncation_psi=1, truncation_cutoff=None, update_emas=False, **synthesis_kwargs):
        # Compute RGB features, density for arbitrary 3D coordinates. Mostly used for extracting shapes. 
        if self.load_lms:
            v, lms = v[:, :5023], v[:, 5023:]
        ws = self.mapping(z, c, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff, update_emas=update_emas)
        batch_size = ws.shape[0]
        eg3d_ws, texture_ws = ws[:, :self.texture_backbone.num_ws], ws[:, self.texture_backbone.num_ws:]
        textures = self.texture_backbone.synthesis(texture_ws, update_emas=update_emas, **synthesis_kwargs)
        # rasterize to three orthogonal views
        rendering_views = [
            [0, 0, 0],
            [0, 90, 0],
            [0, -90, 0],
            [90, 0, 0]
        ]
        rendering_images, alpha_images, uvcoords_images, lm2ds = self.rasterize(v, lms, textures, rendering_views, batch_size, ws.device)
        # generate front mouth masks
        rendering_image_front = rendering_images[0]
        mouths_mask = self.gen_mouth_mask(lm2ds[0])
        rendering_mouth = [rendering_image_front[i:i+1, :][:, :, m[0]:m[1], m[2]:m[3]] for i, m in enumerate(mouths_mask)]
        rendering_mouth = torch.cat([torch.nn.functional.interpolate(uv, size=(64, 64), mode='bilinear', antialias=True) for uv in rendering_mouth], 0)
        
        # generate mouth front plane and integrate back to face front plane
        mouths_plane = self.mouth_backbone.synthesis(rendering_mouth, eg3d_ws, update_emas=update_emas, **synthesis_kwargs)
        rendering_stitch = []
        for rendering, m, mouth_plane in zip(rendering_image_front, mouths_mask, mouths_plane):
            rendering = rendering.unsqueeze(0)
            dummy = torch.zeros_like(rendering)
            dummy[:, :] = rendering
            dummy[:, :, m[0]:m[1], m[2]:m[3]] = torch.nn.functional.interpolate(mouth_plane.unsqueeze(0), size=(m[1]-m[0], m[1]-m[0]), mode='bilinear', antialias=True)
            rendering_stitch.append(dummy)
        rendering_stitch = torch.cat(rendering_stitch, 0)
        rendering_stitch = self.neural_blending.synthesis(rendering_stitch, eg3d_ws, update_emas=update_emas, **synthesis_kwargs)

        # generate static triplane
        static_plane = self.backbone.synthesis(eg3d_ws, update_emas=update_emas, **synthesis_kwargs)
        static_plane = static_plane.view(len(static_plane), 3, 32, static_plane.shape[-2], static_plane.shape[-1])
        
        # blend features of neural texture and tri-plane
        alpha_image = torch.cat(alpha_images, 1).unsqueeze(2)
        rendering_stitch = torch.cat((rendering_stitch, rendering_images[1], rendering_images[2]), 1)
        rendering_stitch = rendering_stitch.view(*static_plane.shape)
        blended_planes = rendering_stitch * alpha_image + static_plane * (1 - alpha_image)

        return self.renderer.run_model(blended_planes, self.decoder, coordinates, directions, self.rendering_kwargs)

    def sample_mixed(self, coordinates, directions, ws, v, truncation_psi=1, truncation_cutoff=None, update_emas=False, **synthesis_kwargs):
        # Same as sample, but expects latent vectors 'ws' instead of Gaussian noise 'z'
        if self.load_lms:
            v, lms = v[:, :5023], v[:, 5023:]
        batch_size = ws.shape[0]
        eg3d_ws, texture_ws = ws[:, :self.texture_backbone.num_ws], ws[:, self.texture_backbone.num_ws:]
        textures = self.texture_backbone.synthesis(texture_ws, update_emas=update_emas, **synthesis_kwargs)
        
        # rasterize to three orthogonal views
        rendering_views = [
            [0, 0, 0],
            [0, 90, 0],
            [0, -90, 0],
            [90, 0, 0]
        ]
        rendering_images, alpha_images, uvcoords_images, lm2ds = self.rasterize(v, lms, textures, rendering_views, batch_size, ws.device)
        
        # generate front mouth masks
        rendering_image_front = rendering_images[0]
        mouths_mask = self.gen_mouth_mask(lm2ds[0])
        rendering_mouth = [rendering_image_front[i:i+1, :][:, :, m[0]:m[1], m[2]:m[3]] for i, m in enumerate(mouths_mask)]
        rendering_mouth = torch.cat([torch.nn.functional.interpolate(uv, size=(64, 64), mode='bilinear', antialias=True) for uv in rendering_mouth], 0)
        
        # generate mouth front plane and integrate back to face front plane
        mouths_plane = self.mouth_backbone.synthesis(rendering_mouth, eg3d_ws, update_emas=update_emas, **synthesis_kwargs)
        rendering_stitch = []
        for rendering, m, mouth_plane in zip(rendering_image_front, mouths_mask, mouths_plane):
            rendering = rendering.unsqueeze(0)
            dummy = torch.zeros_like(rendering)
            dummy[:, :] = rendering
            dummy[:, :, m[0]:m[1], m[2]:m[3]] = torch.nn.functional.interpolate(mouth_plane.unsqueeze(0), size=(m[1]-m[0], m[1]-m[0]), mode='bilinear', antialias=True)
            rendering_stitch.append(dummy)
        rendering_stitch = torch.cat(rendering_stitch, 0)
        rendering_stitch = self.neural_blending.synthesis(rendering_stitch, eg3d_ws, update_emas=update_emas, **synthesis_kwargs)

        # generate static triplane
        static_plane = self.backbone.synthesis(eg3d_ws, update_emas=update_emas, **synthesis_kwargs)
        static_plane = static_plane.view(len(static_plane), 3, 32, static_plane.shape[-2], static_plane.shape[-1])
        
        # blend features of neural texture and tri-plane
        alpha_image = torch.cat(alpha_images, 1).unsqueeze(2)        
        rendering_stitch = torch.cat((rendering_stitch, rendering_images[1], rendering_images[2]), 1)
        rendering_stitch = rendering_stitch.view(*static_plane.shape)
        blended_planes = rendering_stitch * alpha_image + static_plane * (1 - alpha_image)

        return self.renderer.run_model(blended_planes, self.decoder, coordinates, directions, self.rendering_kwargs)

    def forward(self, z, c, v, truncation_psi=1, truncation_cutoff=None, neural_rendering_resolution=None, update_emas=False, cache_backbone=False, use_cached_backbone=False, **synthesis_kwargs):
        # Render a batch of generated images.
        ws = self.mapping(z, c, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff, update_emas=update_emas)
        return self.synthesis(ws, c, v, update_emas=update_emas, neural_rendering_resolution=neural_rendering_resolution, cache_backbone=cache_backbone, use_cached_backbone=use_cached_backbone, **synthesis_kwargs)

    def gen_mouth_mask(self, lms2d):
        lm = lms2d.clone().cpu().numpy() # lms2d: (4, 68, 2)
        lm[..., 0] = lm[..., 0] * 128 + 128
        lm[..., 1] = lm[..., 1] * 128 + 128
        lm_mouth_outer = lm[:, 48:60]  # left-clockwise

        mouth_left = lm_mouth_outer[:, 0]
        mouth_right = lm_mouth_outer[:, 6]
        mouth_avg = (mouth_left + mouth_right) * 0.5 # (4, 2)
        ups, bottoms = np.max(lm_mouth_outer[..., 0], axis=1, keepdims=True), np.min(lm_mouth_outer[..., 0], axis=1, keepdims=True)
        lefts, rights = np.min(lm_mouth_outer[..., 1], axis=1, keepdims=True), np.max(lm_mouth_outer[..., 1], axis=1, keepdims=True)
        mask_res = np.max(np.concatenate((ups - bottoms, rights - lefts), axis=1), axis=1, keepdims=True) * 1.2
        mask_res = mask_res.astype(int)
        mouth_mask = np.concatenate([(mouth_avg[:, 1:] - mask_res//2).astype(int), (mouth_avg[:, 1:] + mask_res//2).astype(int), (mouth_avg[:, 0:1] - mask_res//2).astype(int), (mouth_avg[:, 0:1] + mask_res//2).astype(int)], 1) # (4, 4)
        return mouth_mask

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
