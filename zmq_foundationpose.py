import numpy as np
import cv2

import zmq
import os
import sys
import numpy as np
import argparse
import time
import json
import torch
from torch.utils.data import DataLoader
import open3d as o3d
from PIL import Image
from scipy.spatial.transform import Rotation as R, Slerp
from datetime import datetime

from enum import IntEnum
import time

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


class NetProto_Network(IntEnum):
    FOUNDATION_POSE = 0
    SOM = 1
    GPT = 2
    GRASP_NET = 3

def send_and_wait(port, flag, name):
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.connect(f"tcp://localhost:{port}")
    socket.send(bytes([int(flag)]))

    print(f"[{name}] 명령 전송됨, 응답 대기 중...")
    start_time = time.time()
    dots = ""

    while True:
        try:
            message = socket.recv(zmq.NOBLOCK)
            elapsed = time.time() - start_time
            print(f"\n[{name}] 응답 수신 완료! 대기 시간: {elapsed:.2f}초")
            break
        except zmq.Again:
            time.sleep(0.2)
            print(f"\r[{name}] 대기 중 ... {int(time.time() - start_time)}초 경과", end="", flush=True)


if __name__ == '__main__':

    # object_name = "white and yellow box."
    # model_path =  "/media/vision/data_4TB/grasp/graspnet/models/001/textured.obj" 

    object_name = "the red can."
    model_path =  "/media/vision/data_4TB/grasp/graspnet/models/002/textured.obj" 

    # object_name = "the mustard bottle."
    # model_path =  "/media/vision/data_4TB/grasp/graspnet/models/003/textured.obj" 


    # object_name = "the spam can."
    # model_path =  "/media/vision/data_4TB/grasp/graspnet/models/004/textured.obj" 

    # object_name = "red cracker box."
    # model_path =  "/media/vision/data_4TB/grasp/graspnet/models/000/textured.obj" 

    object_config_data = {
      "object_name": object_name,
      "model_path": model_path,
    }

    file_path  = '/home/vision/packages/FoundationPose/result/object_config.json'
    with open(file_path, "w") as json_file:
        json.dump(object_config_data, json_file, indent=4)


    

    # pipeline = rs.pipeline()
    # config = rs.config()

    # config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    # config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)
    
    # profile = pipeline.start(config)

    # align_to = rs.stream.color
    # align = rs.align(align_to)

    # color_stream = profile.get_stream(rs.stream.color)
    # intrinsics = color_stream.as_video_stream_profile().get_intrinsics()

    # K = np.array([[intrinsics.fx, 0, intrinsics.ppx],
    #                         [0, intrinsics.fy, intrinsics.ppy],
    #                             [0, 0, 1]]).astype(np.float32)
    # print("K : ", K)


    # 실행 예시
    send_and_wait(1111, NetProto_Network.FOUNDATION_POSE, "FOUNDATION_POSE")