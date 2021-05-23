from models.lbs import lbs

def prep_param(pose, exp, shape, cam):
    bsize = pose.size(0)
    shape = shape.expand(bsize, -1)
    return pose, exp, shape, cam

def post_param(pose, exp, shape, cam):
    pass

config = {
    # FLAME
    'model_name': 'flame',
    'flame_model_path': '../GIF/FLAME/models/generic_model.pkl',  # acquire it from FLAME project page
    'cam_prior': 5,
    'flame_lmk_embedding_path': './data/landmark_embedding.npy',
    'tex_space_path': '../FLAME_texture.npz',  # acquire it from FLAME project page
    'camera_params': 6,
    'shape_params': 100,
    'expression_params': 50,
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
    'w_shape_reg': 1e-4,
    'w_expr_reg': 1e-4,
    'w_pose_reg': 0,
    'trim_path': "./data/trim_verts_face.npz",
    'param2verts': lbs,
    'prep_param': prep_param, 
    'post_param': post_param
}
