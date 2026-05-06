# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


from estimater import *
from datareader import *
import argparse
import pyrealsense2 as rs
import numpy as np
import cv2
import sys
sys.path.append('/home/vision/packages/Grounded-Segment-Anything')
from PIL import Image
import matplotlib.pyplot as plt

import time
from datetime import datetime
import yaml
# 필요한 파일 import
import grounded_sam_demo as gs  # 예: some_file.py에서 정의된 함수나 클래스 사용

# sys.path.append(os.path.join("/home/vision/packages/Grounded-Segment-Anything", "GroundingDINO"))
# sys.path.append(os.path.join("/home/vision/packages/Grounded-Segment-Anything", "segment_anything"))


# Grounding DINO - SAM1
import GroundingDINO.groundingdino.datasets.transforms as T
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap

# segment anythingremove_statistical_outlier
from segment_anything import (
    sam_model_registry,
    sam_hq_model_registry,
    SamPredictor
)


# Grounded-SAM-2 관련 파라미터
sys.path.append('/home/vision/packages/Grounded-SAM-2')
import supervision as sv
import pycocotools.mask as mask_util
from pathlib import Path
from supervision.draw.color import ColorPalette
from utils.supervision_utils import CUSTOM_COLOR_MAP
from PIL import Image
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection 

# environment settings
# use bfloat16
# torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

with torch.autocast("cuda", dtype=torch.bfloat16):
  if torch.cuda.is_available() and torch.cuda.get_device_properties(0).major >= 8:
      # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
      torch.backends.cuda.matmul.allow_tf32 = True
      torch.backends.cudnn.allow_tf32 = True
  
  GROUNDING_MODEL = "IDEA-Research/grounding-dino-tiny"
  SAM2_CHECKPOINT = "/home/vision/packages/Grounded-SAM-2/checkpoints/sam2.1_hiera_large.pt"
  SAM2_MODEL_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"
  DEVICE = "cuda"
  
  
  # build SAM2 image predictor
  sam2_checkpoint = SAM2_CHECKPOINT
  model_cfg = SAM2_MODEL_CONFIG
  sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=DEVICE)
  sam2_predictor = SAM2ImagePredictor(sam2_model)
  
  # build grounding dino from huggingface
  model_id = GROUNDING_MODEL
  processor = AutoProcessor.from_pretrained(model_id)
  grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(DEVICE)

##########################################################3



TIME = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_path = f"./logs/{TIME}"
if not os.path.exists(log_path):
    os.makedirs(log_path)

ARM_ID = None 
LEFT = 0
RIGHT = 1

# 리얼센스 카메라 관련 변수 선언
# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()

# Get device product line for setting a supporting resolution
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()
device_product_line = str(device.get_info(rs.camera_info.product_line))

config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

align_to = rs.stream.color
align = rs.align(align_to)
# Start streaming
profile = pipeline.start(config)
sq = 0


depth_sensor = profile.get_device().first_depth_sensor()
unit_value = depth_sensor.get_option(rs.option.depth_units) # 기본 0.001
# print(unit_value)

# depth_sensor.set_option(rs.option.depth_units, 0.001) # l515 사용시 주석

unit_value = depth_sensor.get_option(rs.option.depth_units) # 기본 0.001
print(unit_value)
depth_scale = depth_sensor.get_depth_scale()
# time.sleep(1)  # 5초 동안 코드 실행 일시정지

user_input = None

def get_realsense_intrinsic():
  profile = pipeline.get_active_profile()
  color_profile = rs.video_stream_profile(profile.get_stream(rs.stream.color))
  intr = color_profile.get_intrinsics()
  
  print(intr)

  return intr

def get_mesh_path(obj_name):
   
   name = obj_name.lower()
   
   mesh_file = None
   if "box" in name:
      mesh_file = "/media/vision/data_4TB/grasp/graspnet/models/001/textured.obj"

   elif ("metallic" in name or "cylinder" in name or "cylinderical" in name or "white" in name or "silver" in name or "gray" in name) and ("object" in name or "can" in name):
      mesh_file = "/media/vision/data_4TB/grasp/graspnet/models/monster_2/000/nontextured.obj"

   elif "bottle" in name and ("plastic" in name or "orange" in name or 'white' in name or 'yellow' not in name):
      mesh_file = "/media/vision/data_4TB/grasp/graspnet/models/cola/000/nontextured.obj"

   elif ("round" in name or "apple" in name or 'furit' in name) and "red" in name: #"orange" in name or 
      mesh_file = "/media/vision/data_4TB/grasp/graspnet/models/012/textured.obj"

   elif ("round" in name or "orange" in name) and "orange" in name: #"orange" in name or 
      mesh_file = "/media/vision/data_4TB/grasp/graspnet/models/016/textured.obj"

   elif ("can" in name or "spam" in name) and ("blue" in name or "yellow" in name): 
      mesh_file = "/media/vision/data_4TB/grasp/graspnet/models/004/textured.obj"

   elif "yellow" in name or "bottle" in name or "mustard" in name:
      mesh_file = "/media/vision/data_4TB/grasp/graspnet/models/003/textured.obj"

   elif "red" in name or "can" in name:
      mesh_file = "/media/vision/data_4TB/grasp/graspnet/models/002/textured.obj"

   elif "red" in name and "box" in name:
      mesh_file = "/media/vision/data_4TB/grasp/graspnet/models/000/textured.obj"
      
   print("Object name : ", name , "/mesh_file : ",mesh_file)
   return mesh_file

def get_realsense_data():
#   frames = pipeline.wait_for_frames()
#   aligned_frames = align.process(frames)
#   color_frame = aligned_frames.get_color_frame()
#   aligned_depth_frame = aligned_frames.get_depth_frame()
  
  color = cv2.imread("result/color.png")
  depth = cv2.imread("result/depth.png",-1) 
  
  
  
  if len(depth.shape) == 3: # detph image가 3채널(RGB)로 되어 있는 경우 16비트로 변환
      # This is encoded depth image, let's convert
      depth = np.uint16(depth[:, :, 1]*256) + np.uint16(depth[:, :, 2]) # NOTE: RGB is actually BGR in opencv
      depth = depth.astype(np.uint16) 
  elif len(depth.shape) == 2 and depth.dtype == 'uint16': # 따로 변환하지 않고 그대로 가져옴
      print("no conversion depth is 16bit")
      depth = depth

# #   # L515
#   depth = depth.astype(np.uint32)  # 먼저 uint3211로 변환
#   depth = depth / 4.0  # 이제 4를 곱해도 오버플로우가 발생하지 않음
#   depth = depth.astype(np.uint16)  # 계산 후 uint16으로 다시 변환


  color = np.asanyarray(color)
  depth = np.asanyarray(depth)

  cv2.imwrite(f'./result/color.png', color)
  cv2.imwrite(f'./result/depth.png', depth)

  cv2.imwrite(f'./logs/{TIME}/color.png', color)
  cv2.imwrite(f'./logs/{TIME}/depth.png', depth)
  
  color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)

  return color, depth/1e3

def save_mask_data(output_dir, mask, box, label, image, depth):

    value = 0 # for background

    # mask_img = torch.zeros(mask_list.shape[-2:])
    mask_img = torch.full(mask.shape[-2:], 255)

    masked_input_image = np.zeros_like(image)
    print(masked_input_image.shape)

    mask_img[mask == True] = value + 1
    # 입력 이미지에서 마스크 영역만 남기기
    for c in range(3):  # 입력 이미지가 RGB일 경우
        masked_input_image[:, :, c][mask[0] == True] = image[:, :, c][mask[0] == True]


    
    cv2.imwrite(os.path.join(output_dir, f'mask.png'), mask_img.numpy().astype(np.uint8))
    cv2.imwrite(f'./logs/{TIME}/mask.png', mask_img.numpy().astype(np.uint8))
    
    cv2.imwrite(os.path.join(output_dir, f'masked_color.png'), cv2.cvtColor(masked_input_image,cv2.COLOR_BGR2RGB))

def show_mask(mask, ax, random_color=False):
    
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        # color = np.array([30/255, 144/255, 255/255, 0.6])  # 파랑색
        color = np.array([255/255, 30/255, 30/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_box(box, ax, label):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='red', facecolor=(0,0,0,0), lw=4))
    # ax.text(x0, y0, label)

def get_mask_click_and_crop(color, depth):
  # cfg
  config_file = "/home/vision/packages/Grounded-Segment-Anything/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"  # change the path of the model config file
  grounded_checkpoint = "/home/vision/packages/Grounded-Segment-Anything/groundingdino_swint_ogc.pth"  # change the path of the model
  sam_version = "vit_h"
  sam_checkpoint = "/home/vision/packages/Grounded-Segment-Anything/sam_vit_h_4b8939.pth"
  box_threshold = 0.2
  text_threshold = 0.2
  bert_base_uncased_path = None
  device = "cuda"
  sam_hq_checkpoint = None
  use_sam_hq = False
  
  image, depth = color, depth
  
  # 마우스로 selecting 하는 부분
  global is_selecting
  
  while(True):
    cv2.namedWindow("image")
    cv2.setMouseCallback("image", click_and_crop)
  
    tmp_frame = color[:,:,::-1].copy()
    if len(ref_pt) == 2:
        cv2.rectangle(tmp_frame, ref_pt[0], ref_pt[1], (0,255,0))
    
    cv2.imshow('image', tmp_frame)   
    key = cv2.waitKey(1)  
    if (len(ref_pt) == 3):
        ref_pt[1] = ref_pt[2]
        break
    elif key == ord('r'):
        is_selecting = False
        ref_pt.clear()

  cv2.destroyAllWindows()
  

  image_pil = Image.fromarray(image)
    
  # Apply transformation
  transform = T.Compose(
      [
          T.RandomResize([800], max_size=1333),
          T.ToTensor(),
          T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
      ]
  )
  image,_ = transform(image_pil, None)  # 3, H, W 형식
  
  boxes_filt = np.array((ref_pt[0], ref_pt[1]), dtype=np.float32)

  boxes_filt = torch.tensor(boxes_filt).flatten()
  logging.info(f'boxes_filt :{boxes_filt}')


  global ARM_ID
  if (ref_pt[0][0]+ref_pt[1][0])/2.0 > 320:
      ARM_ID = LEFT
  else:
      ARM_ID = RIGHT
    

  # initialize SAM
  if use_sam_hq:
      predictor = SamPredictor(sam_hq_model_registry[sam_version](checkpoint=sam_hq_checkpoint).to(device))
  else:
      predictor = SamPredictor(sam_model_registry[sam_version](checkpoint=sam_checkpoint).to(device))

  predictor.set_image(color)
  size = image_pil.size
  H, W = size[1], size[0]
  
  
#   for i in range(boxes_filt.size(0)):
#       boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
#       boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
#       boxes_filt[i][2:] += boxes_filt[i][:2]
#   boxes_filt = boxes_filt * torch.Tensor([W, H, W, H])
#   boxes_filt[:2] -= boxes_filt[2:] / 2
#   boxes_filt[2:] += boxes_filt[:2]

  boxes_filt = boxes_filt.cpu()
  transformed_boxes = predictor.transform.apply_boxes_torch(boxes_filt, color.shape[:2]).to(device)
  mask, _, _ = predictor.predict_torch(
      point_coords = None,
      point_labels = None,
      boxes = transformed_boxes.to(device),
      multimask_output = False,
  )

  selected_mask = None
  logging.info(f'mask shape:{mask.shape}')


  mask_np = mask.cpu().numpy().squeeze()  
  if mask_np.ndim != 2:
      raise ValueError(f"Expected mask to be 2D, but got shape {mask_np.shape}")

  # Create a visual overlay
  overlay = color.copy()
  overlay[mask_np == 1] = [0, 255, 0]  # Green mask
  # mask = mask.cpu().numpy()
  # print(mask.shape)
  # exit(0)
  h, w = mask.shape[-2:]
  # mask_image = mask.reshape(h, w, 1) * overlay.reshape(1, 1, -1)

  # Display the image with the mask using matplotlib
  plt.figure(figsize=(8, 6))
  plt.imshow(overlay)
  plt.title(f"Mask {i+1}")
  plt.gcf().canvas.mpl_connect('key_press_event', on_key)
  plt.axis('off')
  plt.show()


  if user_input == 'y':  # User selects this mask
    selected_mask = mask_np
    # 선택한 mask 정보 저장
    output_dir = '/home/vision/packages/FoundationPose/result'
    # draw output image
    plt.figure(figsize=(10, 10))
    plt.imshow(color)
    show_box(boxes_filt.numpy(), plt.gca(), None)
          
    
    plt.axis('off')
    plt.savefig(
        os.path.join(output_dir, f"grounded_sam_output_box.jpg"),
        bbox_inches="tight", dpi=300, pad_inches=0.0
    )
    
    show_mask(mask.cpu().numpy(), plt.gca(), random_color=True)
          
    
    plt.axis('off')
    plt.savefig(
        os.path.join(output_dir, f"grounded_sam_output.jpg"),
        bbox_inches="tight", dpi=300, pad_inches=0.0
    )
    
    # save_mask_data(output_dir, mask, boxes_filt, None, color, depth)
  elif user_input == 'n':
    logging.info(f'Mask Selection Fail !!!')
    selected_mask = None
    

  if selected_mask is not None:
      print("Selected mask saved.")
  else:
      print("No mask selected.")

  # # draw output image
  # for mask in masks:
  #     gs.show_mask(mask.cpu().numpy(), plt.gca(), random_color=True)
  # for box, label in zip(boxes_filt, pred_phrases):
  #     gs.show_box(box.numpy(), plt.gca(), label)  

  return selected_mask

def draw_box_opencv(image_bgr, box, color=(0, 0, 255), thickness=4):
    """
    image_bgr: BGR 이미지 (OpenCV 기본)
    box: [x0, y0, x1, y1]
    """
    x0, y0, x1, y1 = map(int, box)
    img_box = image_bgr.copy()
    cv2.rectangle(img_box, (x0, y0), (x1, y1), color, thickness)
    return img_box


def draw_mask_opencv(image_bgr, mask, color=(30, 30, 255), alpha=0.6):
    """
    image_bgr: BGR 이미지
    mask: 2D (H, W), 0/1 또는 bool
    color: BGR 색상
    alpha: mask 투명도 (0~1)
    """
    img_mask = image_bgr.copy()

    if mask.dtype != np.bool_:
        mask_bool = mask.astype(bool)
    else:
        mask_bool = mask

    overlay = img_mask.copy()
    overlay[mask_bool] = (
        overlay[mask_bool] * (1 - alpha) + np.array(color, dtype=np.float32) * alpha
    ).astype(np.uint8)

    # overlay 자체가 이미 섞인 결과라면 addWeighted는 생략 가능
    return overlay


def draw_box_and_mask_opencv(image_bgr, box, mask, box_color=(0, 0, 255), mask_color=(30, 30, 255), alpha=0.5):
    """
    bbox + mask 둘 다 그린 최종 이미지
    """
    img = draw_mask_opencv(image_bgr, mask, color=mask_color, alpha=alpha)
    img = draw_box_opencv(img, box, color=box_color, thickness=4)
    return img


def get_mask_grounded(color, depth, text_prompt):
  # cfg
  config_file = "/home/vision/packages/Grounded-Segment-Anything/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"  # change the path of the model config file
  grounded_checkpoint = "/home/vision/packages/Grounded-Segment-Anything/groundingdino_swint_ogc.pth"  # change the path of the model
  sam_version = "vit_h"
  sam_checkpoint = "/home/vision/packages/Grounded-Segment-Anything/sam_vit_h_4b8939.pth"
  text_prompt = text_prompt
#   text_prompt = "the spam can"
#   text_prompt = "the yellow box"
#   text_prompt = "the mustard bottle"
  box_threshold = 0.2
  text_threshold = 0.2
  bert_base_uncased_path = None
  device = "cuda"
  sam_hq_checkpoint = None
  use_sam_hq = False
  
  image, depth = color, depth
  

  image_pil = Image.fromarray(image)
    
  # Apply transformation
  transform = T.Compose(
      [
          T.RandomResize([800], max_size=1333),
          T.ToTensor(),
          T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
      ]
  )
  image,_ = transform(image_pil, None)  # 3, H, W 형식
  
  model = gs.load_model(config_file, grounded_checkpoint, bert_base_uncased_path, device=device)
 
  # run grounding dino model
  boxes_filt, pred_phrases = gs.get_grounding_output(
      model, image, text_prompt, box_threshold, text_threshold, device=device
  )

  logging.info(f'pred_phrases :{pred_phrases}')
  logging.info(f'boxes_filt :{boxes_filt}')

     
  print("dddd")
  # initialize SAM
  if use_sam_hq:
      predictor = SamPredictor(sam_hq_model_registry[sam_version](checkpoint=sam_hq_checkpoint).to(device))
  else:
      predictor = SamPredictor(sam_model_registry[sam_version](checkpoint=sam_checkpoint).to(device))

  predictor.set_image(color)
  size = image_pil.size
  H, W = size[1], size[0]
  
  for i in range(boxes_filt.size(0)):
      boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
      boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
      boxes_filt[i][2:] += boxes_filt[i][:2]
  boxes_filt = boxes_filt.cpu()

  select_index = 0

  for i in range(len(boxes_filt)):
    tmp = cv2.cvtColor(color,cv2.COLOR_RGB2BGR)
    box = (boxes_filt[i]).tolist()
    print(box)
    cv2.rectangle(tmp, (int(box[0]),int(box[1])), (int(box[2]), int(box[3])), (0,255,0))
    
    cv2.imshow('image', tmp)   
    key = cv2.waitKey(0)  

    if (key) == ord('y'):  # User selects this mask
      select_index = i
      break
    elif (key) == ord('n'):
      logging.info(f'NONONO!!!')
      continue
  cv2.destroyAllWindows()

#   exit(0)
  boxes_filt_select = []
  boxes_filt_select.append(boxes_filt[i])
  boxes_filt_select = torch.stack(boxes_filt_select)

#   boxes_filt_select = torch.tensor(boxes_filt_select)
  global ARM_ID
  if (boxes_filt_select[0][0]+boxes_filt_select[0][2])/2.0 > 320:
      ARM_ID = LEFT
      print("LEFT")
  else:
      ARM_ID = RIGHT
      print("RIGHT")


  transformed_boxes = predictor.transform.apply_boxes_torch(boxes_filt_select, color.shape[:2]).to(device)
  masks, _, _ = predictor.predict_torch(
      point_coords = None,
      point_labels = None,
      boxes = transformed_boxes.to(device),
      multimask_output = False,
  )

  selected_mask = None
  logging.info(f'mask shape:{masks.shape}')


#   for i, mask in enumerate(masks):
#     logging.info(f'ith mask :{i}')

#     mask_np = mask.cpu().numpy().squeeze()  
#     if mask_np.ndim != 2:
#         raise ValueError(f"Expected mask to be 2D, but got shape {mask_np.shape}")

#     # Create a visual overlay
#     overlay = color.copy()
#     overlay[mask_np == 1] = [0, 255, 0]  # Green mask
#     # mask = mask.cpu().numpy()
#     # print(mask.shape)
#     # exit(0)
#     h, w = mask.shape[-2:]
#     # mask_image = mask.reshape(h, w, 1) * overlay.reshape(1, 1, -1)

#     # Display the image with the mask using matplotlib
#     plt.figure(figsize=(8, 6))
#     plt.imshow(overlay)
#     plt.title(f"Mask {i+1}: {pred_phrases[i]}")
#     plt.gcf().canvas.mpl_connect('key_press_event', on_key)
#     plt.axis('off')
#     plt.show()


#     if user_input == 'y':  # User selects this mask
#       selected_mask = mask_np
#       # 선택한 mask 정보 저장
#       output_dir = '/home/vision/packages/FoundationPose/result'
#       # draw output image
#       plt.figure(figsize=(10, 10))
#       plt.imshow(color)

#       show_box(boxes_filt[i].numpy(), plt.gca(), pred_phrases[i])
            
      
#       plt.axis('off')
#       plt.savefig(
#           os.path.join(output_dir, f"grounded_sam_output_box.jpg"),
#           bbox_inches="tight", dpi=300, pad_inches=0.0
#       )
      
#       show_mask(mask.cpu().numpy(), plt.gca(), random_color=True)
            
      
#       plt.axis('off')
#       plt.savefig(
#           os.path.join(output_dir, f"grounded_sam_output.jpg"),
#           bbox_inches="tight", dpi=300, pad_inches=0.0
#       )
      
#       save_mask_data(output_dir, mask, boxes_filt[i], pred_phrases[i], color, depth)
#       break

#     elif user_input == 'n':
#       logging.info(f'NONONO!!!')
#       continue
    
  for i, mask in enumerate(masks):
    logging.info(f'ith mask :{i}')

    mask_np = mask.cpu().numpy().squeeze()  
    if mask_np.ndim != 2:
        raise ValueError(f"Expected mask to be 2D, but got shape {mask_np.shape}")

    # Create a visual overlay
    overlay = color.copy()
    overlay[mask_np == 1] = [0, 255, 0]  # Green mask
    # mask = mask.cpu().numpy()
    # print(mask.shape)
    # exit(0)
    h, w = mask.shape[-2:]
    # mask_image = mask.reshape(h, w, 1) * overlay.reshape(1, 1, -1)

    # Display the image with the mask using matplotlib
    plt.figure(figsize=(8, 6))
    plt.imshow(overlay)
    plt.title(f"Mask {i+1}: {pred_phrases[i]}")
    plt.gcf().canvas.mpl_connect('key_press_event', on_key)
    plt.axis('off')
    # plt.show()


    selected_mask = mask_np
    # 선택한 mask 정보 저장
    output_dir = '/home/vision/packages/FoundationPose/result'
    # draw output image
    plt.figure(figsize=(10, 10))
    plt.imshow(color)

    show_box(boxes_filt_select[i].numpy(), plt.gca(), pred_phrases[i])
            
      
    plt.axis('off')
    plt.savefig(
        os.path.join(output_dir, f"grounded_sam_output_box.jpg"),
        bbox_inches="tight", dpi=300, pad_inches=0.0
    )
      
    show_mask(mask.cpu().numpy(), plt.gca(), random_color=True)
            
      
    plt.axis('off')
    plt.savefig(
        os.path.join(output_dir, f"grounded_sam_output.jpg"),
        bbox_inches="tight", dpi=300, pad_inches=0.0
    )
      
    save_mask_data(output_dir, mask, boxes_filt[i], pred_phrases[i], color, depth)



  if selected_mask is not None:
      print("Selected mask saved.")
  else:
      print("No mask selected.")

  # # draw output image
  # for mask in masks:
  #     gs.show_mask(mask.cpu().numpy(), plt.gca(), random_color=True)
  # for box, label in zip(boxes_filt, pred_phrases):
  #     gs.show_box(box.numpy(), plt.gca(), label)  

  return selected_mask

def get_mask_grounded_sam2(color, depth, text_prompt):
  
  with torch.autocast("cuda", dtype=torch.bfloat16):
    # cfg
    text_prompt = text_prompt

    device = "cuda"
    
    image, depth = color, depth
    
    text = text_prompt
    
    parts = re.split(r'[.;]', text_prompt)
    parts = ([p.strip() for p in parts if p.strip()])   # 공백/빈 문자열 제거
    need_count = len(parts)

    image_pil = Image.fromarray(image)
  
    ######## grounded sam2 #######################
#     sam2_predictor.set_image(np.array(color.convert("RGB")))
    image = Image.fromarray(color)
    sam2_predictor.set_image(np.array(image.convert("RGB")))
    
    print(text)
  
    inputs = processor(images=image, text=text, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = grounding_model(**inputs)
    
    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        box_threshold=0.4,
        text_threshold=0.3,
        target_sizes=[image.size[::-1]]
    )
  
    print(results)
    print(len(results[0]['labels']))
    

  
    selected_indices = [] 
    h, w, _ = color.shape
  
    for i in range(len(results[0]["labels"])):
      tmp = cv2.cvtColor(color,cv2.COLOR_RGB2BGR)
      box = (results[0]['boxes'][i]).unsqueeze(0).cpu().numpy()
      print(box)

      confidences = results[0]["scores"][i].unsqueeze(0).cpu().numpy().tolist()
      class_names = results[0]["labels"][i]
      class_ids = np.array([i])                # (1,) shape
      detections = sv.Detections(
          xyxy=box,  # (n, 4)
          class_id=class_ids
      )
  
      """
      Note that if you want to use default color map,
      you can set color=ColorPalette.DEFAULT
      """
      box_annotator = sv.BoxAnnotator(color=ColorPalette.from_hex(CUSTOM_COLOR_MAP))
      annotated_frame = box_annotator.annotate(scene=tmp.copy(), detections=detections)
  
      
      cv2.imshow(f'{class_names}', annotated_frame)   
      key = cv2.waitKey(0)  
  
      if (key) == ord('y'):  # User selects this mask
        selected_indices.append(i)
        if len(selected_indices) >= need_count:
            break 
    
      elif (key) == ord('n'):
        logging.info(f'NONONO!!!')
        continue
    cv2.destroyAllWindows()

    all_selected_masks = {}
    all_box_and_mask_img = color.copy()
    all_box_and_mask_img = cv2.cvtColor(all_box_and_mask_img, cv2.COLOR_RGB2BGR)

    PALETTE_BOX = [
        (0, 0, 255),  # red
        (0, 255, 0),  # soft blue-ish
    ]
    PALETTE_MASK = [
        (30, 30, 255),  # soft red-ish
        (30, 255, 30),  # soft blue-ish
    ]


    
    for i in range(len(selected_indices)):
        select_index = selected_indices[i]
        input_boxes = results[0]["boxes"][select_index].cpu().numpy()
        class_names = results[0]["labels"][i]

        mask, scores, logits = sam2_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_boxes,
            multimask_output=False,
        )
        selected_mask = None
        logging.info(f'mask shape:{mask.shape}')
    
        mask_np = mask.squeeze()  
        if mask_np.ndim != 2:
            raise ValueError(f"Expected mask to be 2D, but got shape {mask_np.shape}")  
        # Create a visual overlay
        overlay = color.copy()
        overlay[mask_np == 1] = [255, 0, 0]  # Green mask

        h, w = mask.shape[-2:]

        selected_mask = mask_np

        output_dir = 0
        # 선택한 mask 정보 저장
        for part_name in parts:
            print("class_names:", repr(class_names))
            print("part_name:", repr(part_name))

            cls_words = class_names.lower().split()
            part_lower = part_name.lower()

            # class_names의 모든 단어가 part_name 안에 포함되면 매칭
            if all(w in part_lower for w in cls_words):
                output_dir = os.path.join(
                    '/home/vision/packages/FoundationPose/result',
                    part_name
                )
                os.makedirs(output_dir, exist_ok=True)

                all_selected_masks[part_name] = {
                    "mask": mask_np,
                    "word": part_name,
                }
                break

        # draw output image
        
        box = results[0]['boxes'][select_index].cpu().numpy().squeeze()  # [x0, y0, x1, y1]
        mask_np = mask.squeeze()  # (H, W)

        # color 가 RGB라면 BGR로 바꿔줘야 함
        img_bgr = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)

        # 1) 박스만 그린 이미지 저장
        img_box = draw_box_opencv(img_bgr, box, color=(0, 0, 255), thickness=4)
        cv2.imwrite(os.path.join(output_dir, "grounded_sam_output_box.png"), img_box)

        # 2) 박스 + 마스크 둘 다 그린 이미지 저장
        img_box_mask = draw_box_and_mask_opencv(
            img_bgr, 
            box, 
            mask_np, 
            box_color=(0, 0, 255), 
            mask_color=(30, 30, 255),  # 빨간계열
            alpha=0.6
        )

        all_box_and_mask_img = draw_box_and_mask_opencv(
            all_box_and_mask_img, 
            box, 
            mask_np, 
            box_color=PALETTE_BOX[i % len(PALETTE_BOX)], 
            mask_color=PALETTE_MASK[i % len(PALETTE_MASK)],  # 빨간계열
            alpha=0.6
        )
        cv2.imwrite(os.path.join(output_dir, "grounded_sam_output.png"), img_box_mask)

        save_mask_data(output_dir, mask, results[0]['boxes'][select_index].cpu().numpy().squeeze(), results[0]["scores"][select_index].cpu().numpy().tolist(), color, depth)
    
        if selected_mask is not None:
            print("Selected mask saved.")
        else:
            print("No mask selected.")
  
  # # draw output image
  # for mask in masks:
  #     gs.show_mask(mask.cpu().numpy(), plt.gca(), random_color=True)
  # for box, label in zip(boxes_filt, pred_phrases):
  #     gs.show_box(box.numpy(), plt.gca(), label)  

    cv2.imwrite(os.path.join("./result/grounded_sam_output.png"), all_box_and_mask_img)

  return all_selected_masks

def on_key(event):
    global user_input
    if event.key == 'y':  # User selects this mask
        user_input = 'y'
        plt.close()
    elif event.key == 'n':  # User skips this mask
        user_input = 'n'
        plt.close()

# Y축 기준 90도 회전 행렬 (4x4 동차 좌표계)
def get_y_rotation_matrix_90():
    angle = np.pi / 2  # 90 degrees in radians
    rotation_matrix = np.array([
        [np.cos(angle), 0, np.sin(angle), 0],
        [0, 1, 0, 0],
        [-np.sin(angle), 0, np.cos(angle), 0],
        [0, 0, 0, 1]
    ])
    return rotation_matrix

def get_x_rotation_matrix_90():
    """
    X축 기준 90도 회전 행렬 (4x4 동차 좌표계).

    Returns:
        np.ndarray: X축 기준 90도 회전 행렬.
    """
    angle = np.pi / 2  # 90 degrees in radians
    rotation_matrix = np.array([
        [1, 0, 0, 0],
        [0, np.cos(angle), -np.sin(angle), 0],
        [0, np.sin(angle), np.cos(angle), 0],
        [0, 0, 0, 1]
    ])
    return rotation_matrix

def get_z_rotation_matrix_90():
    """
    Z축 기준 90도 회전 행렬 (4x4 동차 좌표계).

    Returns:
        np.ndarray: Z축 기준 90도 회전 행렬.
    """
    angle = np.pi / 2  # 90 degrees in radians
    rotation_matrix = np.array([
        [np.cos(angle), -np.sin(angle), 0, 0],
        [np.sin(angle), np.cos(angle), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])
    return rotation_matrix

def get_z_rotation_matrix_minus_90():
    """
    Z축 기준 -90도 회전 행렬 (4x4 동차 좌표계).

    Returns:
        np.ndarray: Z축 기준 -90도 회전 행렬.
    """
    angle = -np.pi / 2  # -90 degrees in radians
    rotation_matrix = np.array([
        [np.cos(angle), -np.sin(angle), 0, 0],
        [np.sin(angle), np.cos(angle), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])
    return rotation_matrix

def get_rotation_matrix(axis, angle):
    """
    주어진 축 (x, y, z)과 각도 (degree) 에 대해 4x4 회전 변환 행렬을 반환하는 함수.
    
    Parameters:
        axis (str): 회전할 축 ('x', 'y', 'z')
        angle (float): 회전 각도 (degree 단위)
    
    Returns:
        np.ndarray: 4x4 회전 변환 행렬 (동차 좌표계)
    """
    angle_rad = np.radians(angle)  # degree -> radian 변환

    if axis == 'x':
        rotation_matrix = np.array([
            [1, 0, 0, 0],
            [0, np.cos(angle_rad), -np.sin(angle_rad), 0],
            [0, np.sin(angle_rad), np.cos(angle_rad), 0],
            [0, 0, 0, 1]
        ])
    elif axis == 'y':
        rotation_matrix = np.array([
            [np.cos(angle_rad), 0, np.sin(angle_rad), 0],
            [0, 1, 0, 0],
            [-np.sin(angle_rad), 0, np.cos(angle_rad), 0],
            [0, 0, 0, 1]
        ])
    elif axis == 'z':
        rotation_matrix = np.array([
            [np.cos(angle_rad), -np.sin(angle_rad), 0, 0],
            [np.sin(angle_rad), np.cos(angle_rad), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
    else:
        raise ValueError("축은 'x', 'y', 'z' 중 하나여야 합니다.")

    return rotation_matrix


def compute_angle_between_vectors(v1, v2):
    v1 = v1 / np.linalg.norm(v1)  # 정규화
    v2 = v2 / np.linalg.norm(v2)  # 정규화
    dot_product = np.clip(np.dot(v1, v2), -1.0, 1.0)  # 수치 오차 방지
    angle = np.arccos(dot_product)  # 라디안 단위
    degree = np.degrees(angle)

    if degree > 90:
        degree = degree - 180
    elif degree < -90:
        degree = degree + 180
    return degree


def visualize_with_open3d(color, depth, cam_K, pose, object_name, mesh_file):
    """
    간단한 Open3D 시각화 함수
    """
    # PointCloud 생성
    xyz_map = depth2xyzmap(depth, cam_K)
    valid = depth > 0.1
    points = xyz_map[valid]
    colors = color[valid] / 255.0

    
    
    cloud = o3d.geometry.PointCloud()

    cloud.points = o3d.utility.Vector3dVector(points)
    cloud.colors = o3d.utility.Vector3dVector(colors)

    # 좌표축 추가
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])


    output_dir = f'/home/vision/packages/FoundationPose/result/{object_name}'
    pose_path = os.path.join(output_dir, "output_pose.npy")
    np.save(pose_path, pose)
    
    config_data = {
              "mesh_path": mesh_file,
              "object_name" : object_name
            }
    file_path  = os.path.join(output_dir, "config.json")
    with open(file_path, "w") as json_file:
        json.dump(config_data, json_file, indent=4)
        
    return 
    o3d.visualization.destory_window()


is_selecting = False
ref_pt = []
def click_and_crop(event, x, y, flags, param):
    global is_selecting
    
    if event == cv2.EVENT_LBUTTONDOWN:
        is_selecting = True
        ref_pt.append([x, y])
    # check to see if the left mouse button was released
    elif event == cv2.EVENT_LBUTTONUP:
        # record the ending (x, y) coordinates and indicate that
        # the cropping operation is finished
        if len(ref_pt) == 2:
            is_selecting = False
            ref_pt.append([x, y])
    elif event == cv2.EVENT_MOUSEMOVE:
        if is_selecting == True:
            if len(ref_pt) == 1:
                ref_pt.append([x, y])
            elif len(ref_pt) == 2:
                ref_pt[1] = [x, y]



if __name__=='__main__':
  parser = argparse.ArgumentParser()
  code_dir = os.path.dirname(os.path.realpath(__file__))
  parser.add_argument('--mesh_file', type=str, default=f'{code_dir}/demo_data/mustard0/mesh/untitled.obj')
  parser.add_argument('--text_prompt', type=str, default='mustard bottle')
  parser.add_argument('--test_scene_dir', type=str, default=f'{code_dir}/demo_data/mustard0')
  parser.add_argument('--est_refine_iter', type=int, default=5)
  parser.add_argument('--track_refine_iter', type=int, default=2)
  parser.add_argument('--debug', type=int, default=1)
  parser.add_argument('--debug_dir', type=str, default=f'{code_dir}/debug')
  parser.add_argument('--num_point', type=int, default=20000, help='Point Number [default: 20000]')
  args = parser.parse_args()

  set_logging_format()
  set_seed(0)
  
  # GPT output 결과를 활용 - 사용하지 않을 경우, text 따로 지정 필요
  print("==== ✅ Check GPT Output ===")
  with open("/home/vision/packages/FoundationPose/result/gpt_output.json", "r") as f:
    data = json.load(f)
  
  num_objects = len(data["Object_list"])
  print("Object count:", num_objects)
  
  object_names = [obj["Object_Details"] for obj in data["Object_list"]]
  print("Object names:", object_names)

  text_prompt = " ".join([name + "." for name in object_names])

  # 다른 모듈을 통해 미리 저장된 이미지 데이터 활용
  with open(os.path.join("result","intr.yaml"), 'r') as f:
    data = yaml.load(f, yaml.FullLoader)

  fx, fy = data['fx'], data['fy']
  cx, cy = data['cx'], data['cy']
  cam_K =  np.array([[data['fx'], 0, data['cx']],
              [0, data['fy'], data['cy']],
              [0, 0, 1]])

  color, depth = get_realsense_data()
  color_all = color.copy()

  # sam2를 활용한 segmentation 마스크
  masks = get_mask_grounded_sam2(color,depth, text_prompt)
  
  if len(masks) == len(object_names):
     print("All Mask is well generated")
  else:
     print("❌️ Error : Some mask is wrong.")
  
  for object_name in object_names:
    print("Pose Estimation : ", object_name)
    mesh_file = get_mesh_path(object_name)
    print("mesh file")

    # 자동으로 mesh 파일 loading / 수동일 경우 mesh_file path 수정 필요
    mesh = trimesh.load(mesh_file, force='mesh')
#     mesh.sample_points_uniformly(number_of_points=args.num_point)

    debug = args.debug
    debug_dir = args.debug_dir
    os.system(f'rm -rf {debug_dir}/* && mkdir -p {debug_dir}/track_vis {debug_dir}/ob_in_cam')

    to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
    bbox = np.stack([-extents/2, extents/2], axis=0).reshape(2,3)


    scorer = ScorePredictor()
    refiner = PoseRefinePredictor()
    glctx = dr.RasterizeCudaContext()
    est = FoundationPose(model_pts=mesh.vertices, model_normals=mesh.vertex_normals, mesh=mesh, scorer=scorer, refiner=refiner, debug_dir=debug_dir, debug=debug, glctx=glctx)
    logging.info("estimator initialization done")



    i = 0
    while True:
      logging.info(f'i:{i}')

      depth = (depth).astype(np.float32)


      H, W = color.shape[:2]
      color = cv2.resize(color, (W,H), interpolation=cv2.INTER_NEAREST)
      depth = cv2.resize(depth, (W,H), interpolation=cv2.INTER_NEAREST)
      # depth[(depth<0.1) | (depth>=np.inf)] = 0
      depth[(depth>=np.inf)] = 0


      print("\n\n cam_K : ", cam_K)
      if i==0:
        
        pose = est.register(K=cam_K, rgb=color, depth=depth, ob_mask=masks[object_name]['mask'], iteration=args.est_refine_iter)

      else:
        pose = est.track_one(rgb=color, depth=depth, K=cam_K, iteration=args.track_refine_iter)

      i += 1

      if i == args.track_refine_iter:
          break

      # os.makedirs(f'{debug_dir}/ob_in_cam', exist_ok=True)
      # np.savetxt(f'{debug_dir}/ob_in_cam/{reader.id_strs[i]}.txt', pose.reshape(4,4))

    if debug>=1:
      center_pose = pose@np.linalg.inv(to_origin)
      bbox_coord = pose@np.linalg.inv(to_origin)
      color_object = color.copy()
      
      vis = draw_posed_3d_box(cam_K, img=color_object, ob_in_cam=center_pose, bbox=bbox)
      vis = draw_xyz_axis(color_object, ob_in_cam=center_pose, scale=0.1, K=cam_K, thickness=3, transparency=0, is_input_rgb=True)

      vis_all = draw_posed_3d_box(cam_K, img=color_all, ob_in_cam=center_pose, bbox=bbox)
      vis_all = draw_xyz_axis(color_all, ob_in_cam=center_pose, scale=0.1, K=cam_K, thickness=3, transparency=0, is_input_rgb=True)

      # 결과물 저장
    #   cv2.imshow('1', vis[...,::-1])
      cv2.imwrite(f'/home/vision/packages/FoundationPose/result/{object_name}/output_pose.png', vis[...,::-1])
      cv2.imwrite(f'/home/vision/packages/FoundationPose/result/output_pose.png', vis_all[...,::-1])
    #   cv2.waitKey(1)  
      visualize_with_open3d(color_object, depth, cam_K, pose, object_name,  mesh_file)
        # cv2.imshow('1', vis[...,::-1])
        # cv2.waitKey(1)


