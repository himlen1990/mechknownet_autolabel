from __future__ import division

from collections import defaultdict

import cv2
import matplotlib.cm
import numpy as np
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont

from srhandnet.data import font_path
from srhandnet.kinematics import normalize_mpii_keypoint_names_to_indices

cmap = matplotlib.cm.get_cmap('hsv')

N_HAND_SEQUENCE = 20
category_trans = int(0.6 * 255)


def visualize_hand_rects(img, bboxes, labels=None, copy=False,
                         font_size=None):
    height, width, _ = img.shape
    long_side = min(width, height)
    font_size = font_size or max(int(round((long_side / 20))), 1)
    box_width = max(int(round(long_side / 180)), 1)
    font = ImageFont.truetype(font_path(), font_size)

    result_vis = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(result_vis)

    box_color = (191, 40, 41)
    for _, (y1, x1, y2, x2) in bboxes:
        draw.rectangle((x1, y1, x2, y2), outline=box_color + (255,),
                       width=box_width)

    bbox_captions = defaultdict(list)
    if labels is not None:
        for i_box, label, color in labels:
            bbox_captions[i_box].append((label, color))

    for i_box, (score, box) in enumerate(bboxes):
        captions = []
        bg_colors = []

        for label, color in bbox_captions[i_box]:
            captions.append(label)
            bg_colors.append(color)

        if score is not None:
            conf = " %.2f" % score
            captions.append('score ' + conf)
            bg_colors.append((176, 85, 234))

        if len(captions) == 0:
            continue
        y1, x1, y2, x2 = box
        overlay = Image.new("RGBA", result_vis.size, (0, 0, 0, 0))
        trans_draw = ImageDraw.Draw(overlay)
        caption_sizes = [trans_draw.textsize(caption, font=font)
                         for caption in captions]
        caption_widths, caption_heights = list(zip(*caption_sizes))
        max_height = max(caption_heights)
        rec_height = int(round(1.8 * max_height))
        space_height = int(round(0.2 * max_height))
        total_height = (rec_height + space_height) \
            * (len(captions) - 1) \
            + rec_height
        width_pad = max(font_size // 2, 1)
        start_y = max(round(y1) - total_height, space_height)

        for i, caption in enumerate(captions):
            r_x1 = round(x1)
            r_y1 = start_y + (rec_height + space_height) * i
            r_x2 = r_x1 + caption_widths[i] + width_pad * 2
            r_y2 = r_y1 + rec_height
            rec_pos = (r_x1, r_y1, r_x2, r_y2)

            height_pad = round((rec_height - caption_heights[i]) / 2)
            text_pos = (r_x1 + width_pad, r_y1 + height_pad)

            trans_draw.rectangle(rec_pos, fill=bg_colors[i]
                                 + (category_trans,))
            trans_draw.text(text_pos, caption,
                            fill=(255, 255, 255, category_trans), font=font,
                            align="center")
        result_vis = Image.alpha_composite(result_vis, overlay)

    pil_img = Image.fromarray(img[..., ::-1])
    pil_img = pil_img.convert("RGBA")
    pil_img = Image.alpha_composite(pil_img, result_vis)
    pil_img = pil_img.convert("RGB")
    if copy:
        return np.array(pil_img, dtype=np.uint8)[..., ::-1]
    else:
        img[:] = np.array(pil_img, dtype=np.uint8)[..., ::-1]
        return img


def visualize_hand_keypoints(img, keypoints, copy=False,
                             hand_indices=None,
                             alpha=1.0,
                             thickness_circle=None):
    if hand_indices is not None:
        hand_indices = normalize_mpii_keypoint_names_to_indices(
            hand_indices)

    if copy:
        img = img.copy()

    canvas = None
    if alpha != 1.0:
        alpha = min(1.0, max(0.0, alpha))
        canvas = img.copy()

    H, W, _ = img.shape

    # parameters for drawing
    if thickness_circle is None:
        thickness_circle_ratio = 1. / 120.
        thickness_circle = max(int(np.sqrt(H * W)
                                   * thickness_circle_ratio + 0.5), 2)

    for _, keypoint_index, _, (y, x) in keypoints:
        if hand_indices is not None:
            if keypoint_index not in hand_indices:
                continue
        rgba = np.array(cmap(1. * keypoint_index / N_HAND_SEQUENCE))
        color = rgba[:3] * 255
        if canvas is not None:
            cv2.circle(canvas, (int(x), int(y)), thickness_circle, color, -1)
        else:
            cv2.circle(img, (int(x), int(y)), thickness_circle, color, -1)

    if canvas is not None:
        cv2.addWeighted(img, 1.0 - alpha, canvas, alpha, 0, img)

    return img


def visualize_text(img, text, copy=False):
    pil_img = Image.fromarray(img[..., ::-1])
    pil_img = pil_img.convert("RGBA")

    height, width, _ = img.shape
    long_side = min(width, height)
    font_size = max(int(round((long_side / 10))), 1)
    font = ImageFont.truetype(font_path(), font_size)

    overlay = Image.new("RGBA", pil_img.size, (0, 0, 0, 0))
    trans_draw = ImageDraw.Draw(overlay)
    text_width, text_height = trans_draw.textsize(text, font=font)
    width_pad = max(font_size // 2, 1)
    rec_height = int(round(1.8 * text_height))
    height_pad = round((rec_height - text_height) / 2)

    r_x1 = 0
    r_y2 = rec_height
    r_x2 = r_x1 + text_width + width_pad * 2
    r_y1 = 0
    rec_pos = (r_x1, r_y1, r_x2, r_y2)
    text_pos = (r_x1 + width_pad, height_pad)

    trans_draw.rectangle(rec_pos, fill=(0, 0, 0, category_trans))
    trans_draw.text(text_pos, text, fill=(255, 255, 255, category_trans),
                    font=font, align="center")

    pil_img = Image.alpha_composite(pil_img, overlay)

    pil_img = pil_img.convert("RGB")

    if copy:
        return np.array(pil_img)[..., ::-1]
    else:
        img[:] = np.array(pil_img)[..., ::-1]
        return img
