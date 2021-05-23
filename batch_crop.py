import pdb
from tqdm import tqdm
import numpy as np
import torch
from face_seg_model import BiSeNet
from face_alignment.detection import sfd_detector as detector
import cv2
import os
import sys
import util
from wj_fitting import resize_para, face_seg

def crop_img(ori_image, rect, cropped_size):
    l, t, r, b = rect
    center_x = r - (r - l) // 2
    center_y = b - (b - t) // 2
    w = (r - l) * 1.2
    h = (b - t) * 1.2
    crop_size = max(w, h)
    if crop_size > cropped_size:
        crop_ly = int(max(0, center_y - crop_size // 2))
        crop_lx = int(max(0, center_x - crop_size // 2))
        crop_ly = int(min(ori_image.shape[0] - crop_size, crop_ly))
        crop_lx = int(min(ori_image.shape[1] - crop_size, crop_lx))
        crop_image = ori_image[crop_ly: int(crop_ly + crop_size), crop_lx: int(crop_lx + crop_size), :]
    else:

        crop_ly = int(max(0, center_y - cropped_size // 2))
        crop_lx = int(max(0, center_x - cropped_size // 2))
        crop_ly = int(min(ori_image.shape[0] - cropped_size, crop_ly))
        crop_lx = int(min(ori_image.shape[1] - cropped_size, crop_lx))
        crop_image = ori_image[crop_ly: int(crop_ly + cropped_size), crop_lx: int(crop_lx + cropped_size), :]
    new_rect = [l - crop_lx, t - crop_ly, r - crop_lx, b - crop_ly]
    return crop_image, new_rect

config = {
    'face_seg_model': './model/face_seg.pth',
    'seg_class': 19,
    'face_detect_type': "2D",
    'rect_model_path': "./model/s3fd.pth",
    'rect_thresh': 0.5,
    'landmark_model_path': "./model/2DFAN4-11f355bf06.pth.tar",
}
config = util.dict2obj(config)
cropped_size = 256

device_name = "cuda"
device = "cuda:0"
path_out = "../ssddata/FaceWarehouse_Data_0_cropseg"
path_in = "../ssddata/FaceWarehouse_Data_0_raw/raw"
img_listfile = "../ssddata/fwh_imagelist.txt"
with open(img_listfile, "r") as f:
    names = f.readlines()
names = [l.strip("\n") for l in names]
img_paths = []
for fname in names:
    src_img_path = os.path.join(path_in, fname)
    img_paths.append((fname.replace(".png", ""),src_img_path))
    prefix = "/".join(fname.split("/")[:-1])
    if not os.path.exists(os.path.join(path_out, prefix)):
        os.makedirs(os.path.join(path_out, prefix))

img = cv2.imread(img_paths[0][1])
w_h_scale = resize_para(img)

face_detect = detector.SFDDetector(device_name, config.rect_model_path, w_h_scale)
seg_net = BiSeNet(n_classes=config.seg_class)
seg_net.cuda()
seg_net.load_state_dict(torch.load(config.face_seg_model))
seg_net.eval()

for path in tqdm(img_paths):
    fname = path[0]
    img = cv2.imread(path[1])
    w_h_scale = resize_para(img)
    face_detect.w, face_detect.h, face_detect.input_scale = w_h_scale

    images = []
    image_masks = []
    bbox = face_detect.extract(img, config.rect_thresh)
    assert len(bbox)>0
    crop_image, new_bbox = crop_img(img, bbox[0], cropped_size)
    image = cv2.resize(crop_image, (cropped_size, cropped_size)).astype(np.float32) / 255.
    image = image[:, :, [2, 1, 0]].transpose(2, 0, 1)
    images.append(torch.from_numpy(image[None, :, :, :]).double().to(device))
    image_mask = face_seg(crop_image, seg_net)
    image_mask = cv2.resize(image_mask, (cropped_size, cropped_size))
    image_mask = image_mask[..., None].astype('float32')
    image_mask = image_mask.transpose(2, 0, 1)
    image_mask_bn = np.zeros_like(image_mask)
    image_mask_bn[np.where(image_mask != 0)] = 1.
    image_masks.append(torch.from_numpy(image_mask_bn[None, :, :, :]).double().to(device))

    #images = torch.cat(images, dim=0)
    #image_masks = torch.cat(image_masks, dim=0)
    #torch.save(image_masks, f"{path_out}/{fname}.pth")
    image_mask = (image_mask[0]*255).astype(np.uint8)
    image = (image.transpose((1,2,0))*255).astype(np.uint8)
    cv2.imwrite(f"{path_out}/{fname}.png", image[:,:,::-1])
    cv2.imwrite(f"{path_out}/{fname}_mask.png", image_mask)
