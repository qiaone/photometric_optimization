from models.lbs import blbs
import pdb
import torch

def prep_param(pose, exp, shape, cam):
    #pose = torch.cat((pose, torch.zeros_like(pose)), 1)
    exp = torch.cat((torch.ones_like(exp[:,:1]), exp), 1)
    return pose, exp, shape, cam

def post_param(pose, exp, shape, cam):
    exp.data = torch.clamp(exp.data, 0, 1)

config = {
    # FLAME
    'model_name':'fwh',
    #'flame_model_path': './data/generic_model.pkl',  # acquire it from FLAME project page
    'flame_model_path': '../GIF/FLAME/models/generic_model.pkl',  # acquire it from FLAME project page
    'bs_model_path': './data/fwh_corealign_50_47_2flame.npy',  # acquire it from FLAME project page
    'shape_factors_path': './data/fwh_factorsalign_150_50_2flame.npy',
    'cam_prior': 0.5, 

    'flame_lmk_embedding_path': './data/landmark_embedding.npy',
    #'tex_space_path': './data/FLAME_texture.npz',  # acquire it from FLAME project page
    'tex_space_path': '../FLAME_texture.npz',  # acquire it from FLAME project page
    'camera_params': 6,
    'shape_params': 50,
    'expression_params': 46,
    'pose_params': 3,
    'tex_params': 50,
    'use_face_contour': True,

    'cropped_size': 256,
    'batch_size': 1,
    'image_size': 224,
    'e_lr': 0.005,
    'e_wd': 0.0001,
    'savefolder': './test_results/',
    # weights of losses and reg terms
    'w_pho': 8,
    'w_lmks': 1,
    'w_shape_reg': 1e-1,
    'w_expr_reg': 1e-2,
    'w_pose_reg': 0,
    'trim_path': "./data/trim_verts_face.npz",
    'param2verts': blbs,
    'prep_param': prep_param, 
    'post_param': post_param,
    'prep_param_rigid': prep_param, 
    'post_param_rigid': post_param
}
