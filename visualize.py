# Copyright (c) 2015 Lightricks. All rights reserved.
from data import COLORS
from utils import timer
from layers.output_utils import postprocess

from data import cfg
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2


def show_sample(img, gt, masks):
    dets = {}
    dets['class'] = gt[:, -1]
    dets['box'] = gt[:, :4]
    dets['score'] = np.ones(len(gt))
    dets['mask'] = masks
    det_out = {}
    det_out['net'] = 'coco-gt'
    det_out['detection'] = dets
    if img.max() > 1:
        img = img / 255.0
    img1 = prep_display(dets, img)
    if img1 is not None:
        plt.imshow(img1[...,::-1])
        plt.show()

from collections import defaultdict
color_cache = defaultdict(lambda: {})
def prep_display(dets_out, img, class_color=False, mask_alpha=0.45):
    """
    Note: If undo_transform=False then im_h and im_w are allowed to be None.
    """

    _, h, w = img.shape

    with timer.env('Postprocess'):
        save = cfg.rescore_bbox
        cfg.rescore_bbox = True

        classes = dets_out['class']
        boxes = dets_out['box']
        scores = dets_out['score']
        masks = dets_out['mask']
        t = classes, scores, boxes, masks
        # t = postprocess([dets_out], w, h, visualize_lincomb=False,
        #                 crop_masks=False,
        #                 score_threshold=0)
        cfg.rescore_bbox = save

    with timer.env('Copy'):
        idx = t[1].argsort(0)[:5]

        if cfg.eval_mask_branch:
            # Masks are drawn on the GPU, so don't copy
            masks = t[3][idx]
        classes, scores, boxes = [x[idx] for x in t[:3]]

    num_dets_to_consider = min(5, classes.shape[0])
    for j in range(num_dets_to_consider):
        if scores[j] < 0:
            num_dets_to_consider = j
            break

    # Quick and dirty lambda for selecting the color for a particular index
    # Also keeps track of a per-gpu color cache for maximum speed
    def get_color(j, on_gpu=None):
        global color_cache
        color_idx = (classes[j] * 5 if class_color else j * 5) % len(COLORS)

        if on_gpu is not None and color_idx in color_cache[on_gpu]:
            return color_cache[on_gpu][color_idx]
        else:
            color = COLORS[color_idx]
            if True:
                # The image might come in as RGB or BRG, depending
                color = (color[2], color[1], color[0])
            if on_gpu is not None:
                color = torch.Tensor(color).to(on_gpu).float() / 255.
                color_cache[on_gpu][color_idx] = color
            return color

    # First, draw the masks on the GPU where we can do it really fast
    # Beware: very fast but possibly unintelligible mask-drawing code ahead
    # I wish I had access to OpenGL or Vulkan but alas, I guess Pytorch tensor operations will have to suffice
    if cfg.eval_mask_branch and num_dets_to_consider > 0:
        # After this, mask is of size [num_dets, h, w, 1]
        masks = masks[:num_dets_to_consider, :, :, None]

        # Prepare the RGB images for each mask given their color (size [num_dets, h, w, 1])
        colors = torch.cat([get_color(j, on_gpu='cpu').view(1, 1, 1, 3) for j in
                            range(num_dets_to_consider)], dim=0)
        masks = torch.from_numpy(masks)
        masks_color = masks.repeat(1, 1, 1, 3) * colors * mask_alpha

        # This is 1 everywhere except for 1-mask_alpha where the mask is
        inv_alph_masks = masks * (-mask_alpha) + 1

        # I did the math for this on pen and paper. This whole block should be equivalent to:
        #    for j in range(num_dets_to_consider):
        #        img_gpu = img_gpu * inv_alph_masks[j] + masks_color[j]
        masks_color_summand = masks_color[0]
        if num_dets_to_consider > 1:
            inv_alph_cumul = inv_alph_masks[:(num_dets_to_consider - 1)].cumprod(dim=0)
            masks_color_cumul = masks_color[1:] * inv_alph_cumul
            masks_color_summand += masks_color_cumul.sum(dim=0)

        img = img.transpose(0, 2).transpose(0, 1)
        img_gpu = img * inv_alph_masks.prod(dim=0) + masks_color_summand

    else:
        return None
    # Then draw the stuff that needs to be done on the cpu
    # Note, make sure this is a uint8 tensor or opencv will not anti alias text for whatever reason
    img_numpy = (img_gpu * 255).byte().cpu().numpy()

    if num_dets_to_consider == 0:
        return img_numpy

    if True:
        for j in reversed(range(num_dets_to_consider)):
            x1, y1, x2, y2 = boxes[j, :]
            x1, x2 = int(w * x1), int(w * x2)
            y1, y2 = int(h * y1), int(h * y2)
            color = get_color(j)
            score = scores[j]

            if True:
                cv2.rectangle(img_numpy, (x1, y1), (x2, y2), color, 1)

            if True:
                _class = cfg.dataset.class_names[int(classes[j])]
                text_str = '%s: %.2f' % (_class, score) if False else _class

                font_face = cv2.FONT_HERSHEY_DUPLEX
                font_scale = 0.6
                font_thickness = 1

                text_w, text_h = cv2.getTextSize(text_str, font_face, font_scale, font_thickness)[0]

                text_pt = (x1, y1 - 3)
                text_color = [255, 255, 255]

                cv2.rectangle(img_numpy, (x1, y1), (x1 + text_w, y1 - text_h - 4), color, -1)
                cv2.putText(img_numpy, text_str, text_pt, font_face, font_scale, text_color,
                            font_thickness, cv2.LINE_AA)

    return img_numpy