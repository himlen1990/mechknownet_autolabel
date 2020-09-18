from __future__ import division

from collections import namedtuple
import os.path as osp

import cv2
import gdown
import numpy as np
import torch
import torch.nn as nn

from srhandnet.peak import peak_local_max

download_dir = osp.expanduser('~/.srhandnet')

TRAIN_IMAGE_HEIGHT = 256
TRAIN_IMAGE_WIDTH = 256


def _get_srhand_pretrained_model():
    gdown.cached_download(
        url='https://drive.google.com/uc?id=16Jg8HhaFaThzFSbWbEixMLE3SAnOzvzL',
        path=osp.join(download_dir, 'SRHandNet.pts'),
        md5='9f39d3baa43cf1c962c8f752c009eb14',
        quiet=True,
    )
    return osp.join(download_dir, 'SRHandNet.pts')


Bbox = namedtuple('Bbox', ('y1', 'x1', 'y2', 'x2'))
HandRect = namedtuple('HandRect', ('score', 'bbox'))


class SRHandNet(nn.Module):

    def __init__(self, threshold=0.2):
        super(SRHandNet, self).__init__()
        self.model = torch.jit.load(_get_srhand_pretrained_model(),
                                    map_location=torch.device('cpu'))
        self.threshold = threshold

    def _get_device_type(self):
        return next(self.parameters()).device

    def transform_net_input(self, input_tensor, src_img, tensor_index=0):
        H, W, _ = src_img.shape
        ratio = min(float(input_tensor.size(2)) / H,
                    float(input_tensor.size(3)) / W)
        M = np.array([[ratio, 0, 0],
                      [0, ratio, 0]], dtype=np.float32)
        dst = np.zeros(
            (int(input_tensor.size(3)), int(input_tensor.size(2)), 3),
            dtype=src_img.dtype)
        cv2.warpAffine(
            src_img, M,
            (int(input_tensor.size(3)), int(input_tensor.size(2))),
            dst,
            cv2.INTER_CUBIC, cv2.BORDER_CONSTANT,
            (128, 128, 128))
        dst = np.array(dst, dtype=np.float32)
        dst = dst / 255. - 0.5
        input_tensor[tensor_index][:] = torch.from_numpy(
            dst.transpose(2, 0, 1))
        return ratio

    def detect_bbox(self, img):
        """Return hand bounding boxes

        Parameters
        ----------
        img : numpy.ndarray
            shape (H, W, 3)

        Returns
        -------
        handrect : list(numpy.ndarray)
            bounding box of hand regions.
        """
        H, W, _ = img.shape
        input_tensor = torch.zeros(
            (1, 3, TRAIN_IMAGE_HEIGHT, TRAIN_IMAGE_WIDTH))

        # transform the input data
        ratio_input_to_net = self.transform_net_input(input_tensor, img)

        input_tensor = input_tensor.to(self._get_device_type())

        # run the network
        with torch.no_grad():
            heatmap = self.model(input_tensor)[3]

        # copy the 3-channel rect map
        ratio_net_downsample = TRAIN_IMAGE_HEIGHT / float(heatmap.size(2))
        rect_map_idx = heatmap.size(1) - 3
        handrect = []

        scores, coords = peak_local_max(
            heatmap[:1, rect_map_idx:rect_map_idx + 1], self.threshold)

        kernel_size = 5
        ap = nn.AvgPool2d(kernel_size=kernel_size,
                          padding=(kernel_size - 1) // 2,
                          stride=1)
        avg_img = ap(heatmap[:1, rect_map_idx + 1:])

        if len(coords) == 0:
            return handrect
        ratio_wh = torch.clamp(
            avg_img[:, :, coords[:, 2], coords[:, 3]][0], 0, 1)

        for score, (pos_y, pos_x), (ratio_width, ratio_height) in zip(
                scores.cpu(),
                coords[:, 2:].cpu(),
                ratio_wh.T.cpu()):
            points = (pos_y * ratio_net_downsample / ratio_input_to_net,
                      pos_x * ratio_net_downsample / ratio_input_to_net)
            rect_w = ratio_width * TRAIN_IMAGE_WIDTH / ratio_input_to_net
            rect_h = ratio_height * TRAIN_IMAGE_HEIGHT / ratio_input_to_net
            l_t_x = points[1] - rect_w / 2.
            l_t_y = points[0] - rect_h / 2.
            r_b_x = points[1] + rect_w / 2.
            r_b_y = points[0] + rect_h / 2.

            l_t_x = max(l_t_x, 0.)
            l_t_y = max(l_t_y, 0.)
            r_b_x = min(r_b_x, W - 1.)
            r_b_y = min(r_b_y, H - 1.)
            handrect.append(
                HandRect(score,
                         Bbox(float(l_t_y), float(l_t_x),
                              float(r_b_y), float(r_b_x))))
        return handrect

    def create_tensor(self, img, handrect):
        input_tensor = torch.zeros(
            (len(handrect), 3, TRAIN_IMAGE_HEIGHT, TRAIN_IMAGE_WIDTH))
        ratio_input_to_net = np.zeros(len(handrect), dtype=np.float32)

        for i, (_, rect) in enumerate(handrect):
            y1, x1, y2, x2 = map(int, rect)
            ratio_input_to_net[i] = self.transform_net_input(
                input_tensor, img[y1:y2, x1:x2], i)
        return input_tensor

    def detect_hand_keypoints(self, img, handrect, return_heatmaps=False):
        if len(handrect) == 0:
            if return_heatmaps:
                return [], []
            else:
                return []
        hand_points = []
        # transform the input data and copy to the gpu
        input_tensor = torch.zeros(
            (len(handrect), 3, TRAIN_IMAGE_HEIGHT, TRAIN_IMAGE_WIDTH))
        ratio_input_to_net = np.zeros(len(handrect), dtype=np.float32)

        for i, (_, rect) in enumerate(handrect):
            y1, x1, y2, x2 = map(int, rect)
            ratio_input_to_net[i] = self.transform_net_input(
                input_tensor, img[y1:y2, x1:x2], i)

        input_tensor = input_tensor.to(self._get_device_type())

        # run the network
        with torch.no_grad():
            heatmaps = self.model(input_tensor)[3]

        intensities, coords = peak_local_max(
            heatmaps[:, :heatmaps.size(1) - 3], self.threshold)

        # determine the joint position
        ratio_net_downsample = TRAIN_IMAGE_HEIGHT / float(heatmaps.size(2))
        coords = coords.cpu()
        intensities = intensities.cpu()
        for value, (rect_index, i, pos_y, pos_x) in zip(intensities, coords):
            y = pos_y * ratio_net_downsample / ratio_input_to_net[rect_index] \
                + handrect[rect_index].bbox.y1
            x = pos_x * ratio_net_downsample / ratio_input_to_net[rect_index] \
                + handrect[rect_index].bbox.x1
            hand_points.append((int(rect_index), int(i), float(value),
                                (float(y), float(x))))
        if return_heatmaps:
            return hand_points, heatmaps
        return hand_points

    def pyramid_inference(self, img, handrect=None, return_bbox=False):
        handrect = handrect or []
        H, W, _ = img.shape
        if len(handrect) == 0:
            handrect = self.detect_bbox(img)
            if len(handrect) == 0:
                if return_bbox:
                    return [], []
                else:
                    return []

        # for each small image detect the joint points
        keypoints = self.detect_hand_keypoints(img, handrect)

        if return_bbox is False:
            return keypoints
        else:
            return keypoints, handrect
