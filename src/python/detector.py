#!/usr/bin/env python3

import argparse
import cv2
import json
import numpy as np
import pandas as pd
import os
import PIL.Image, PIL.ImageDraw, PIL.ImageFont
import time
import torch
import torch2trt
import torchvision.transforms as transforms
import trt_pose.coco
import trt_pose.models

from collections import deque
from torch2trt import TRTModule
from trt_pose.draw_objects import DrawObjects
from trt_pose.parse_objects import ParseObjects

class BreathRateDetector(object):
    def __init__(self):
        topology = None
        with open('/trt_pose/tasks/human_pose/human_pose.json', 'r') as f:
            human_pose = json.load(f)
            topology = trt_pose.coco.coco_category_to_topology(human_pose)

        self._topology = topology

        self._MODEL_WEIGHTS = '/trt_pose/tasks/human_pose/resnet18_baseline_att_224x224_A_epoch_249.pth'
        self._OPTIMIZED_MODEL = '/trt_pose/tasks/human_pose/resnet18_baseline_att_224x224_A_epoch_249_trt.pth'
        self._num_parts = len(human_pose['keypoints'])
        self._num_links = len(human_pose['skeleton'])
        print('BreathRateDetector: using resnet model')
        self._model = trt_pose.models.resnet18_baseline_att(self._num_parts, 2 * self._num_links).cuda().eval()
        self._WIDTH = 224
        self._HEIGHT = 224
        self._data = torch.zeros((1, 3, self._HEIGHT, self._WIDTH)).cuda()

        if os.path.exists(self._OPTIMIZED_MODEL) == False:
            print('BreathRateDetector: -- Converting TensorRT models. This may takes several minutes...')
            self._model.load_state_dict(torch.load(self._MODEL_WEIGHTS))
            model_trt = torch2trt.torch2trt(self._model, [self._data], fp16_mode=True, max_workspace_size=1<<25)
            torch.save(model_trt.state_dict(), self._OPTIMIZED_MODEL)
            print('BreathRateDetector: -- Conversion complete --')

        print('BreathRateDetector: loading TRT model.')
        self._model_trt = TRTModule()
        self._model_trt.load_state_dict(torch.load(self._OPTIMIZED_MODEL))

        self._mean = torch.Tensor([0.485, 0.456, 0.406]).cuda()
        self._std = torch.Tensor([0.229, 0.224, 0.225]).cuda()
        self._device = torch.device('cuda')

        self._parse_objects = ParseObjects(topology)
        self._draw_objects = DrawObjects(topology)

    def get_keypoint(self, humans, hnum, peaks):
        #check invalid human index
        kpoint = []
        human = humans[0][hnum]
        C = human.shape[0]
        for j in range(C):
            k = int(human[j])
            if k >= 0:
                peak = peaks[0][j][k]   # peak[1]:width, peak[0]:height
                peak = (j, float(peak[0]), float(peak[1]))
                kpoint.append(peak)
                #print('index:%d : success [%5.3f, %5.3f]'%(j, peak[1], peak[2]) )
            else:    
                peak = (j, None, None)
                kpoint.append(peak)
                #print('index:%d : None %d'%(j, k) )
        return kpoint

    def preprocess(self, image):
        self._device = torch.device('cuda')
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = PIL.Image.fromarray(image)
        image = transforms.functional.to_tensor(image).to(self._device)
        image.sub_(self._mean[:, None, None]).div_(self._std[:, None, None])
        return image[None, ...]

    def execute(self, img):
        img = cv2.resize(img, dsize=(self._WIDTH, self._HEIGHT), interpolation=cv2.INTER_AREA)
        preprocessed = self.preprocess(img)
        cmap, paf = self._model_trt(preprocessed)
        cmap, paf = cmap.detach().cpu(), paf.detach().cpu()
        counts, objects, peaks = self._parse_objects(cmap, paf)#, cmap_threshold=0.15, link_threshold=0.15)
        key_points = []
        for i in range(counts[0]):
            #print("Human index:%d "%( i ))
            kpoint = self.get_keypoint(objects, i, peaks)
            key_points.extend(kpoint)
        return key_points

    def draw_point(self, draw, w, h, key, idx1, idx2, fill=(51, 51, 204), offset=0):
        thickness = 5
        if (offset == 0 and all(key[idx1]) and all(key[idx2])) or (offset==1 and all(key[idx1][offset:]) and all(key[idx2])):
            draw.line([ round(key[idx1][2] * w), round(key[idx1][1] * h), round(key[idx2][2] * w), round(key[idx2][1] * h)],width = thickness, fill=fill)

    def draw_keypoints(self, img, key):
        pilimg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pilimg = PIL.Image.fromarray(pilimg)
        w, h = pilimg.size
        draw = PIL.ImageDraw.Draw(pilimg)
        #draw Rankle -> RKnee (16-> 14)
        self.draw_point(draw, w, h, key, 16, 14)
        #draw RKnee -> Rhip (14-> 12)
        self.draw_point(draw, w, h, key, 14, 12)
        #draw Rhip -> Lhip (12-> 11)
        self.draw_point(draw, w, h, key, 12, 11)
        #draw Lhip -> Lknee (11-> 13)
        self.draw_point(draw, w, h, key, 11, 13)
        #draw Lknee -> Lankle (13-> 15)
        self.draw_point(draw, w, h, key, 13, 15)

        #draw Rwrist -> Relbow (10-> 8)
        self.draw_point(draw, w, h, key, 10, 8, fill=(255,255,51))
        #draw Relbow -> Rshoulder (8-> 6)
        self.draw_point(draw, w, h, key, 8, 6, fill=(255,255,51))
        #draw Rshoulder -> Lshoulder (6-> 5)
        self.draw_point(draw, w, h, key, 6, 5, fill=(255,255,0))
        #draw Lshoulder -> Lelbow (5-> 7)
        self.draw_point(draw, w, h, key, 5, 7, fill=(51,255,51))
        #draw Lelbow -> Lwrist (7-> 9)
        self.draw_point(draw, w, h, key, 7, 9, fill=(51,255,51))

        #draw Rshoulder -> RHip (6-> 12)
        self.draw_point(draw, w, h, key, 6, 12, fill=(153,0,51))
        #draw Lshoulder -> LHip (5-> 11)
        self.draw_point(draw, w, h, key, 5, 11, fill=(153,0,51))

        #draw nose -> Reye (0-> 2)
        self.draw_point(draw, w, h, key, 0, 2, fill=(219,0,219), offset=1)
        #draw Reye -> Rear (2-> 4)
        self.draw_point(draw, w, h, key, 2, 4, fill=(219,0,219))
        #draw nose -> Leye (0-> 1)
        self.draw_point(draw, w, h, key, 0, 1, fill=(219,0,219), offset=1)
        #draw Leye -> Lear (1-> 3)
        self.draw_point(draw, w, h, key, 1, 3, fill=(219,0,219))
        #draw nose -> neck (0-> 17)
        self.draw_point(draw, w, h, key, 0, 17, fill=(255,255,0), offset=1)

        cv2img = np.asarray(pilimg, dtype="uint8")
        cv2img = cv2.cvtColor(cv2img, cv2.COLOR_BGR2RGB)
        return cv2img

    def rate_from_keypoints(self, keypoints):
        rshoulder_y_series = [x[6][1] for x in keypoints]
        lshoulder_y_series = [x[5][1] for x in keypoints]

        frame_count = len(keypoints)

        sum_series = []
        for i in range(frame_count):
            cur = np.nan
            if rshoulder_y_series[i] is not None and lshoulder_y_series[i] is not None:
                cur = rshoulder_y_series[i] + lshoulder_y_series[i]
            sum_series.append(cur)
        interp_series = pd.Series(sum_series).interpolate().tolist()
        tr = np.absolute(np.fft.fft(interp_series))
        cycles_in_period = 1+np.argmax(tr[1:30])

        brpm = cycles_in_period * (600. / frame_count)

        return brpm
          

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Detect breath rate from the camera or a video source.')
    parser.add_argument('-c', '--camera_id', metavar='CAMERA-ID', type=str, help='The numeric id of the video capture device.', default=None)
    parser.add_argument('-v', '--video_file', metavar='VIDEO-FILE', type=str, help='Path to a video file on which we should run detection.', default=None)
    args = parser.parse_args()

    cap = None
    frame_delay = 1
    target_fps = 10
    skip_frames = 0

    if args.camera_id and args.video_file:
        parser.error('Only supply one of CAMERA-ID or VIDEO-FILE.')
    elif not args.camera_id and not args.video_file:
        parser.error('Must supply either CAMERA-ID or VIDEO-FILE.')
    elif args.camera_id:
        print('Using video device ID {}'.format(args.camera_id))
        cap = cv2.VideoCapture(int(args.camera_id))
        cap.set(cv2.CAP_PROP_FPS, target_fps)
        fps = target_fps
    else:
        print('Reading from video file: {}'.format(args.video_file))
        cap = cv2.VideoCapture(args.video_file)
        fps = round(cap.get(cv2.CAP_PROP_FPS))
        skip_frames = int(fps / target_fps) - 1
    
    frame_delay = int(1000 / fps)
    print('Using frame_delay of {} ({} fps), skipping {}'.format(frame_delay, fps, skip_frames))

    detector = BreathRateDetector()

    cur_skip = 0
    point_buffer = deque([])
    pb_start = time.time()
    pb_frames = 0

    start_ms = time.time() * 1000
    while(cap.isOpened()):
        ret, frame = cap.read()
        pb_frames = pb_frames + 1

        if (skip_frames > 0):
            if cur_skip != skip_frames:
                cur_skip = cur_skip + 1
                continue
            else:
                cur_skip = 0

        if frame is not None:
            points = detector.execute(frame)
            if len(points) > 0:
                frame = detector.draw_keypoints(frame, points)
            point_buffer.append(points)

            cv2.imshow('frame',frame)
            now = time.time() * 1000
            ellapsed = now - start_ms
            start_ms = now
            wait_ms = max(frame_delay - int(ellapsed), 1)
            if cv2.waitKey(wait_ms) & 0xFF == ord('q'):
                break
        else:
            break
    pb_end = time.time()
    pb_sec = pb_end - pb_start
    print('Read {} frames in {} seconds ({} fps)'.format(pb_frames, pb_sec, pb_frames/pb_sec))

    cap.release()
    cv2.destroyAllWindows()

    rate = detector.rate_from_keypoints(point_buffer)
    print('rate: {}'.format(rate))