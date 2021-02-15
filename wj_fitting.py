import os, sys
import cv2
import torch
import torchvision
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import datetime
from face_seg_model import BiSeNet
import torchvision.transforms as transforms
from renderer import Renderer
import util
from PIL import Image
from face_alignment.detection import sfd_detector as detector
from face_alignment.detection import FAN_landmark
import matplotlib.pyplot as plt

from photometric_fitting import PhotometricFitting

torch.backends.cudnn.benchmark = True


class WJPhotometricFitting(PhotometricFitting):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def crop_img(self, ori_image, rect):
        l, t, r, b = rect
        center_x = r - (r - l) // 2
        center_y = b - (b - t) // 2
        w = (r - l) * 1.2
        h = (b - t) * 1.2
        crop_size = max(w, h)
        cropped_size = self.config.cropped_size
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

    def run(self, img, net, rect_detect, landmark_detect, rect_thresh, save_name, savefolder):
        # The implementation is potentially able to optimize with images(batch_size>1),
        # here we show the example with a single image fitting
        images = []
        landmarks = []
        image_masks = []
        bbox = rect_detect.extract(img, rect_thresh)
        if len(bbox) > 0:
            crop_image, new_bbox = self.crop_img(img, bbox[0])
            #plt.imshow(crop_image)
            #plt.show()
            #pdb.set_trace()

            resize_img, landmark = landmark_detect.extract([crop_image, [new_bbox]])
            landmark = landmark[0]
            landmark[:, 0] = landmark[:, 0] / float(resize_img.shape[1]) * 2 - 1
            landmark[:, 1] = landmark[:, 1] / float(resize_img.shape[0]) * 2 - 1
            landmarks.append(torch.from_numpy(landmark)[None, :, :].double().to(self.device))

            image = cv2.resize(crop_image, (self.config.cropped_size, self.config.cropped_size)).astype(np.float32) / 255.
            image = image[:, :, [2, 1, 0]].transpose(2, 0, 1)
            images.append(torch.from_numpy(image[None, :, :, :]).double().to(self.device))
            image_mask = face_seg(crop_image, net)
            image_mask = cv2.resize(image_mask, (self.config.cropped_size, self.config.cropped_size))
            image_mask = image_mask[..., None].astype('float32')
            image_mask = image_mask.transpose(2, 0, 1)
            image_mask_bn = np.zeros_like(image_mask)
            image_mask_bn[np.where(image_mask != 0)] = 1.
            image_masks.append(torch.from_numpy(image_mask_bn[None, :, :, :]).double().to(self.device))

            images = torch.cat(images, dim=0)
            images = F.interpolate(images, [self.image_size, self.image_size])
            image_masks = torch.cat(image_masks, dim=0)
            image_masks = F.interpolate(image_masks, [self.image_size, self.image_size])

            landmarks = torch.cat(landmarks, dim=0)
            util.check_mkdir(savefolder)
            save_name = os.path.join(savefolder, save_name)
            images = images.float()
            landmarks = landmarks.float()

            # optimize
            single_params = self.optimize(images, landmarks, image_masks, savefolder)
            self.render.save_obj(filename=save_name,
                                 vertices=torch.from_numpy(single_params['verts'][0]).to(self.device),
                                 textures=torch.from_numpy(single_params['albedos'][0]).to(self.device)
                                 )
            np.save(save_name, single_params)


def face_seg(img, net):
    face_area = [1, 2, 3, 4, 5, 6, 10, 11, 12, 13]
    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    pil_image = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    resize_pil_image = pil_image.resize((512, 512), Image.BILINEAR)
    tensor_image = to_tensor(resize_pil_image)
    tensor_image = torch.unsqueeze(tensor_image, 0)
    tensor_image = tensor_image.cuda()
    out = net(tensor_image)[0]
    parsing = out.squeeze(0).cpu().detach().numpy().argmax(0)
    vis_parsing_anno = parsing.copy().astype(np.uint8)
    vis_parsing_anno_color = np.zeros((vis_parsing_anno.shape[0], vis_parsing_anno.shape[1]))
    num_of_class = np.max(vis_parsing_anno)

    for pi in range(1, num_of_class + 1):
        if pi in face_area:
            index = np.where(vis_parsing_anno == pi)
            vis_parsing_anno_color[index[0], index[1]] = 1

    return vis_parsing_anno_color




def resize_para(ori_frame):
    w, h, c = ori_frame.shape
    d = max(w, h)
    scale_to = 640 if d >= 1280 else d / 2
    scale_to = max(64, scale_to)
    input_scale = d / scale_to
    w = int(w / input_scale)
    h = int(h / input_scale)
    image_info = [w, h, input_scale]
    return image_info


def draw_train_process(title, iters, loss, label_loss):
    plt.title(title, fontsize=24)
    plt.xlabel("iter", fontsize=20)
    plt.ylabel("loss", fontsize=20)
    plt.plot(iters, loss[0], color='red', label=label_loss[0])
    plt.plot(iters, loss[1], color='green', label=label_loss[1])
    plt.plot(iters, loss[2], color='blue', label=label_loss[2])
    plt.legend()
    plt.grid()
    plt.show()


if __name__ == '__main__':
    import pdb
    import importlib
    import torch
    image_path = str(sys.argv[1])
    save_path = str(sys.argv[2])
    device_name = str(sys.argv[3])
    if len(sys.argv)>4:
        model_name = str(sys.argv[4])
    else:
        model_name = 'flame'

    model_filename = "conf." + model_name
    modellib = importlib.import_module(model_filename)
    config = modellib.config
    config_append = {
        'face_seg_model': './model/face_seg.pth',
        'seg_class': 19,
        'face_detect_type': "2D",
        'rect_model_path': "./model/s3fd.pth",
        'rect_thresh': 0.5,
        'landmark_model_path': "./model/2DFAN4-11f355bf06.pth.tar",
    }
    for k,v in config_append.items():
        config[k] = v

    config = util.dict2obj(config)
    #config.savefolder = "./test_results/debug"
    config.savefolder = save_path

    save_name = os.path.split(image_path)[1].split(".")[0] + '.obj'
    util.check_mkdir(config.savefolder)
    fitting = WJPhotometricFitting(config, device=device_name)
    img = cv2.imread(image_path)
    w_h_scale = resize_para(img)

    face_detect = detector.SFDDetector(device_name, config.rect_model_path, w_h_scale)
    face_landmark = FAN_landmark.FANLandmarks(device_name, config.landmark_model_path, config.face_detect_type)

    seg_net = BiSeNet(n_classes=config.seg_class)
    seg_net.cuda()
    seg_net.load_state_dict(torch.load(config.face_seg_model))
    seg_net.eval()

    fitting.run(img, seg_net, face_detect, face_landmark, config.rect_thresh, save_name, config.savefolder)
