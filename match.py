from __future__ import absolute_import, division

import torch
from siamrpn import SiamRPN
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import os
from siamrpn import TrackerSiamRPN


def crop_and_resize(img, center, size, out_size,
                    border_type=cv2.BORDER_CONSTANT,
                    border_value=(0, 0, 0),
                    interp=cv2.INTER_NEAREST):
    # convert box to corners (0-indexed)
    size = round(size)
    corners = np.concatenate((
        np.round(center - (size - 1) / 2),
        np.round(center - (size - 1) / 2) + size))
    corners = np.round(corners).astype(int)

    # pad image if necessary
    pads = np.concatenate((
        -corners[:2], corners[2:] - img.shape[:2]))
    npad = max(0, int(pads.max()))
    if npad > 0:
        img = cv2.copyMakeBorder(
            img, npad, npad, npad, npad,
            border_type, value=border_value)

    # crop image patch
    corners = (corners + npad).astype(int)
    patch = img[corners[0]:corners[2], corners[1]:corners[3]]

    # resize to out_size
    patch = cv2.resize(patch, (out_size, out_size),
                       interpolation=interp)

    return patch


def crop_image(img, th=30, tw=30, f = 2):
    h, w, c = img.shape
    nh = np.arange(int(th/f/2), h - int(th/f/2), int(th/f))
    nw = np.arange(int(tw/f/2), w - int(tw/f/2), int(tw/f))
    x = [crop_and_resize(img, np.array([p, q]), np.sqrt(th * tw),
                         out_size=271,
                         border_value=np.mean(img, axis=(0, 1))) for p in nh.flatten() for q in nw.flatten()]
    for i, xx in enumerate(x):
        cv2.imwrite('test2/%d.png' % (i), xx)

    return x, (len(nh), len(nw))


if __name__ == '__main__':
    # setup tracker
    name = 5
    name = str(name).zfill(2)
    net_path = 'pretrained/siamrpn/model.pth'
    tracker = TrackerSiamRPN(net_path=net_path)

    ori_img = cv2.imread('/DATA2/wxr/pacman/data-png/%s.png'%(name))[:, :, :]
    # ori_img = cv2.imread('/home/wxr/projects/siamrpn-pytorch/test2/17.png')[:, :, :]
    h, w, c = ori_img.shape
    template = cv2.imread('/DATA2/wxr/pacman/template.png')
    template = cv2.resize(template, (127, 127),
                          interpolation=cv2.INTER_NEAREST)
    template = cv2.copyMakeBorder(template, 72, 72, 72, 72, cv2.BORDER_CONSTANT)
    cv2.imwrite('template.png', template)
    imgs, (nh, nw) = crop_image(ori_img, f = 1)
    print(len(imgs), (nh, nw))

    tracker.init(template, None)
    # box, response = tracker.update(ori_img)
    # print(box, response)

    boxes = []
    responses = []
    for img in imgs:
        tracker.init(template, None)
        box, response = tracker.update(img)
        boxes.append(box)
        responses.append(response)

    responses = np.array(responses)
    max_id = np.argmax(responses)
    print('Get max in %d.png, \nResponse = %f,\nbbox ='%(max_id, responses[max_id]), boxes[max_id])
    r_map = np.reshape(responses, (nh, nw))
    r_map = (r_map - np.min(r_map)) / (np.max(r_map) - np.min(r_map)) * 255


    r_map = cv2.resize(r_map, (nw * 30, nh * 30), interpolation=cv2.INTER_NEAREST)
    print(h, w, nw * 30, nh * 30)
    r_map = cv2.copyMakeBorder(r_map, 0, h - nh * 30, 0, w - nw * 30, cv2.BORDER_CONSTANT)
    r_map = cv2.cvtColor(r_map.astype(np.uint8), cv2.COLOR_GRAY2BGR)

    for i in range(nh):
        for j in range(nw):
            idx = i * nw + j
            box = boxes[idx] / 271 * 30
            box = box.astype(np.int)
            x, y, bw, bh = box[0], box[1], box[2] , box[3]
            y1, y2, x1, x2 = i * 30 + y, i * 30 + y + bh, j * 30 + x, j * 30 + x + bw
            r_map[y1:y2, x1:x2, 0:1] -= 80
    added_image = cv2.addWeighted(ori_img, 0.6, r_map, 0.4, 0)
    cv2.imwrite('output/%s.png'%(name), added_image)


