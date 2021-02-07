import pdb
import cv2
from get_full_verts import get_full_verts, image_meshing
import matplotlib.pyplot as plt
import util
from renderer import Pytorch3dRasterizer
import torch
import torch.nn as nn
import numpy as np
import pytorch3d.transforms
from pytorch3d.io import load_obj



class ImageRenderer(nn.Module):
    def __init__(self, image_size, obj_filename, uv_size=256):
        super(ImageRenderer, self).__init__()
        self.image_size = image_size
        self.uv_size = uv_size
        verts, faces, aux = load_obj(obj_filename)
        faces = faces.verts_idx[None, ...]
        self.rasterizer = Pytorch3dRasterizer(image_size)

        # faces
        self.register_buffer('faces', faces)

    def forward(self, cam, vertices, images, cam_new):
        full_vertices, N_bd = get_full_verts(vertices)
        t_vertices = util.batch_orth_proj(full_vertices, cam)
        t_vertices[..., 1:] = - t_vertices[..., 1:]
        t_vertices[...,2] = t_vertices[...,2]+10
        t_vertices = image_meshing(t_vertices, N_bd)
        t_vertices[...,:2] = torch.clamp(t_vertices[...,:2], -1,1)
        t_vertices[:,:,2] =t_vertices[:,:,2]-9
        batch_size = vertices.shape[0]
        ## rasterizer near 0 far 100. move mesh so minz larger than 0
        uvcoords = t_vertices.clone()
        # Attributes
        uvcoords = torch.cat([uvcoords[:,:,:2], uvcoords[:, :, 0:1] * 0. + 1.], -1)  # [bz, ntv, 3]
        face_vertices = util.face_vertices(uvcoords, self.faces.expand(batch_size, -1, -1))
        # render
        attributes = face_vertices.detach()
        full_vertices, N_bd = get_full_verts(vertices)
        transformed_vertices = util.batch_orth_proj(full_vertices, cam_new)
        transformed_vertices[..., 1:] = - transformed_vertices[..., 1:]
        transformed_vertices[...,2] = transformed_vertices[...,2]+10
        transformed_vertices = image_meshing(transformed_vertices, N_bd)
        transformed_vertices[...,:2] = torch.clamp(transformed_vertices[...,:2], -1,1)
        rendering = self.rasterizer(transformed_vertices, self.faces.expand(batch_size, -1, -1), attributes)

        alpha_images = rendering[:, -1, :, :][:, None, :, :].detach()

        # albedo
        uvcoords_images = rendering[:, :3, :, :]
        grid = (uvcoords_images).permute(0, 2, 3, 1)[:, :, :, :2]

        results = F.grid_sample(images, grid, align_corners=False)
        return {'rotate_images':results}



if __name__ == "__main__":
    import torchvision
    import util
    import torch.nn.functional as F

    image_size = 256
    param = np.load("./test_results/00000.npy", allow_pickle=True)[()]
    vertices = torch.Tensor(param['verts0'])
    cam = torch.Tensor(param['cam'])
    images = []
    image = cv2.resize(cv2.imread("./FFHQ/00000.png"), 
            (image_size, image_size)).astype(np.float32) / 255.
    image = image[:, :, [2, 1, 0]].transpose(2, 0, 1)
    images.append(torch.from_numpy(image[None, :, :, :]))
    images = torch.cat(images, dim=0)

    mesh_file = './data/full.obj'
    render = ImageRenderer(image_size, obj_filename=mesh_file)

    angles = torch.Tensor([0, 20, 0])[None,...]/180.0 * np.pi # rotation angles xyz
    cam_new = cam.clone()
    angles = torch.abs(angles)*torch.sign(cam_new[:,:3])
    cam_new[:,:3] = cam_new[:,:3]+angles

    ops = render(cam, vertices, images, cam_new)

    grids = {}
    visind = range(1)  # [0]
    grids['images'] = torchvision.utils.make_grid(images[visind]).detach().cpu()
    grids['rotateimage'] = torchvision.utils.make_grid(
        (ops['rotate_images'])[visind].detach().cpu())
    grid = torch.cat(list(grids.values()), 1)
    grid_image = (grid.numpy().transpose(1, 2, 0).copy() * 255)[:, :, [2, 1, 0]]
    grid_image = np.minimum(np.maximum(grid_image, 0), 255).astype(np.uint8)

    cv2.imwrite('result.jpg', grid_image)
