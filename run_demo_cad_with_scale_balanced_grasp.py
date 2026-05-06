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


# Grounding DINO
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

TIME = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_path = f"./logs/{TIME}"
if not os.path.exists(log_path):
    os.makedirs(log_path)

ARM_ID = None 
LEFT = 0
RIGHT = 1

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
def get_realsense_data():
  frames = pipeline.wait_for_frames()
  aligned_frames = align.process(frames)
  color_frame = aligned_frames.get_color_frame()
  aligned_depth_frame = aligned_frames.get_depth_frame()



  color = np.asanyarray(color_frame.get_data())
  depth = np.asanyarray(aligned_depth_frame.get_data())

#   color = cv2.imread("/media/vision/data_4TB/hyundai/data_0221/6_color.png")
#   depth = cv2.imread("/media/vision/data_4TB/hyundai/data_0221/6_depth.png", -1)

#   color = cv2.imread("/media/vision/data_4TB/hyundai/data_0224/9_color.png")
#   depth = cv2.imread("/media/vision/data_4TB/hyundai/data_0224/9_depth.png", -1)

#   color = cv2.imread("/media/vision/data_4TB/hyundai/data_0224/6_color.png")
#   depth = cv2.imread("/media/vision/data_4TB/hyundai/data_0224/6_depth.png", -1)

#   color = cv2.imread("/media/vision/data_4TB/hyundai/data_0317/4_color.png")
#   depth = cv2.imread("/media/vision/data_4TB/hyundai/data_0317/4_depth.png", -1)
  
#   color = cv2.imread("/media/vision/data_4TB/hyundai/data_test/3_color.png")
#   depth = cv2.imread("/media/vision/data_4TB/hyundai/data_test/3_depth.png", -1)

#   color = cv2.imread("/media/vision/data_4TB/grasp/graspnet/scenes/scene_0000/realsense/rgb/0000.png")
#   depth = cv2.imread("/media/vision/data_4TB/grasp/graspnet/scenes/scene_0000/realsense/depth/0000.png", -1)

#   color = cv2.imread("/media/vision/data_4TB/hyundai/data_0224/14_color.png")
#   depth = cv2.imread("/media/vision/data_4TB/hyundai/data_0224/14_depth.png", -1)
  
# 
#   color = cv2.imread("/media/vision/data_4TB/hyundai/data_0610/7_color.png")
#   depth = cv2.imread("/media/vision/data_4TB/hyundai/data_0610/7_depth.png", -1)

  color = cv2.imread("logs/2025-08-27_16-13-25/color.png")
  depth = cv2.imread("logs/2025-08-27_16-13-25/depth.png",-1) 

#   color = cv2.imread("/media/vision/data_4TB/hyundai/data_0825/15_color.png")
#   depth = cv2.imread("/media/vision/data_4TB/hyundai/data_0825/15_depth.png",-1)
  
  
  cv2.imwrite(f'./logs/{TIME}/color.png', color)
  cv2.imwrite(f'./logs/{TIME}/depth.png', depth)

  if len(depth.shape) == 3: # detph image가 3채널(RGB)로 되어 있는 경우 16비트로 변환
      # This is encoded depth image, let's convert
      depth = np.uint16(depth[:, :, 1]*256) + np.uint16(depth[:, :, 2]) # NOTE: RGB is actually BGR in opencv
      depth = depth.astype(np.uint16) 
  elif len(depth.shape) == 2 and depth.dtype == 'uint16': # 따로 변환하지 않고 그대로 가져옴
      print("no conversion depth is 16bit")
      depth = depth

# #   # L515
  depth = depth.astype(np.uint32)  # 먼저 uint32로 변환
  depth = depth / 4.0  # 이제 4를 곱해도 오버플로우가 발생하지 않음
  depth = depth.astype(np.uint16)  # 계산 후 uint16으로 다시 변환


  color = np.asanyarray(color)
  depth = np.asanyarray(depth)

  cv2.imwrite(f'./result/color.png', color)
  cv2.imwrite(f'./result/depth.png', depth)

  
  
  color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)

  return color, depth/1e3

def save_mask_data(output_dir, mask, box, label, image, depth):

    value = 0 # for background

    # mask_img = torch.zeros(mask_list.shape[-2:])
    mask_img = torch.full(mask.shape[-2:], 255)

    masked_input_image = np.zeros_like(image)
    print(masked_input_image.shape)
    masked_input_depth = np.zeros_like(depth)

    mask_img[mask.cpu().numpy() == True] = value + 1
    mask_array = mask.cpu().numpy()[0]  # 마스크를 numpy 배열로 변환
    # 입력 이미지에서 마스크 영역만 남기기
    for c in range(3):  # 입력 이미지가 RGB일 경우
        masked_input_image[:, :, c][mask.cpu().numpy()[0] == True] = image[:, :, c][mask.cpu().numpy()[0] == True]

    
    # ✅ 깊이 이미지에서 마스크 적용
    masked_input_depth[mask_array] = depth[mask_array]  # 마스크 영역에만 깊이 값 복사

    # maksed_input_depth = depth
    plt.figure(figsize=(10, 10))
    plt.imshow(mask_img.numpy())
    plt.axis('off')
    plt.savefig(os.path.join(output_dir, 'mask.jpg'), bbox_inches="tight", dpi=300, pad_inches=0.0)
    
    cv2.imwrite(os.path.join(output_dir, f'mask.png'), mask_img.numpy().astype(np.uint8))
    cv2.imwrite(f'./logs/{TIME}/mask.png', mask_img.numpy().astype(np.uint8))
    print("\n\n\n\depth shape",masked_input_depth.shape)
    # plt.figure(figsize=(10, 10))
    # plt.imshow(masked_input_image)
    # plt.axis('off')
    # plt.savefig(os.path.join(output_dir, 'mask_output.jpg'), bbox_inches="tight", dpi=300, pad_inches=0.0)
    cv2.imwrite(os.path.join(output_dir, f'masked_color.png'), cv2.cvtColor(masked_input_image,cv2.COLOR_BGR2RGB))

    # plt.figure(figsize=(10, 10))
    # plt.imshow(maksed_input_depth)
    # plt.axis('off')
    # plt.savefig(os.path.join(output_dir, 'mask_output_depth.jpg'), bbox_inches="tight", dpi=300, pad_inches=0.0)
    cv2.imwrite(os.path.join(output_dir, f'masked_depth.png'), masked_input_depth*100.0)

def show_mask(mask, ax, random_color=False):
    
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
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


def visualize_with_open3d(color, depth, cam_K, pose, bbox, bbox_coord, mesh):
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
    
    # trans_mat = np.array([[1,0,0,0],[0,1,0,0],[0,0,-1,0],[0,0,0,1]])
    # cloud.transform(trans_mat)

    # 좌표축 추가
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
    # coordinate_frame.transform(pose)

    # z_flip = np.eye(4)
    # z_flip[2, 2] = -1
    # pose = pose @ z_flip

    # # Y축 기준 90도 회전 행렬 가져오기
    # z_rotation_90 = get_z_rotation_matrix_90()

    # # 기존 pose에 회전 행렬 적용
    # pose = pose @ z_rotation_90

    # 3D 박스 생성
    min_xyz = bbox.min(axis=0)
    max_xyz = bbox.max(axis=0)

    corners = [ 
        [min_xyz[0], min_xyz[1], min_xyz[2]],
        [max_xyz[0], min_xyz[1], min_xyz[2]],
        [max_xyz[0], max_xyz[1], min_xyz[2]],
        [min_xyz[0], max_xyz[1], min_xyz[2]],
        [min_xyz[0], min_xyz[1], max_xyz[2]],
        [max_xyz[0], min_xyz[1], max_xyz[2]],
        [max_xyz[0], max_xyz[1], max_xyz[2]],
        [min_xyz[0], max_xyz[1], max_xyz[2]],
    ]
    corners = np.array(corners)
    transformed_corners = (bbox_coord[:3, :3] @ corners.T).T + bbox_coord[:3, 3]

    lines = [
        [0, 1], [1, 2], [2, 3], [3, 0],  # Bottom face
        [4, 5], [5, 6], [6, 7], [7, 4],  # Top face
        [0, 4], [1, 5], [2, 6], [3, 7],  # Vertical lines
    ]
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(transformed_corners)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector([[0, 1, 0] for _ in range(len(lines))])  # Green


    # # CAD 모델 추가
    mesh_copy = mesh.copy()
    mesh_copy.apply_transform(pose)  # Trimesh 객체 변환
    # X축: 빨간색
        # Y축: 초록색
        # Z축: 파란색

    o3d_mesh = o3d.geometry.TriangleMesh()
    o3d_mesh.vertices = o3d.utility.Vector3dVector(mesh_copy.vertices)
    o3d_mesh.triangles = o3d.utility.Vector3iVector(mesh_copy.faces)
    sampled_pcd = o3d_mesh.sample_points_uniformly(number_of_points=args.num_point)


    cad_points = np.asarray(mesh_copy.vertices)  # CAD 파일의 vertices 가져오기
    cad_center = np.mean(cad_points, axis=0)  # 중심점 계산
    cad_colors = (
        np.asarray(mesh_copy.visual.vertex_colors[:, :3]) / 255.0
        if hasattr(mesh_copy.visual, 'vertex_colors') and mesh_copy.visual.vertex_colors is not None
        else np.full_like(cad_points, [0.5, 0.5, 0.5])  # 기본 회색
    )

    cad_cloud = o3d.geometry.PointCloud()
    cad_cloud.points = o3d.utility.Vector3dVector(cad_points)
    cad_cloud.colors = o3d.utility.Vector3dVector(cad_colors)
    

    
    # translation = pose[:3, :3] @ translation + pose[:3, 3]
    # rotation_matrix = pose[:3, :3] @ rotation_matrix
  
    # 3️⃣ Grasp 좌표축 생성


    #####################################3
    cad_points = np.asarray(mesh_copy.vertices)
    cad_colors = (
        np.asarray(mesh_copy.visual.vertex_colors[:, :3]) / 255.0
        if hasattr(mesh_copy.visual, 'vertex_colors') and mesh_copy.visual.vertex_colors is not None
        else np.full_like(cad_points, [0.5, 0.5, 0.5])  # 기본 회색
    )
    # CAD Point Cloud 저장
    output_dir = '/home/vision/packages/FoundationPose/result'
    cad_points_path = os.path.join(output_dir, "cad_pointcloud.npy")
    np.save(cad_points_path, cad_points)
    # print(f"CAD Point Cloud 저장 완료: {cad_points_path}")

    output_dir = '/home/vision/packages/FoundationPose/result'
    pose_path = os.path.join(output_dir, "output_pose.npy")
    np.save(pose_path, pose)
    # print(f"CAD Point Cloud 저장 완료: {pose_path}")

    
    transform = pose
    file_path = "/home/vision/packages/calibration/data/data_0416_new/no_depth/hand_eye_v1_c2b_nominal_mean.yaml"
    with open(file_path, 'r') as f:
        data = yaml.load(f, yaml.FullLoader)
    cam2base = np.array(data['c2b'])
    cam2base[:3, 3] /= 1000.0
    base2target = cam2base @ transform 
    object_z = base2target[:3,2]
    base_matrix = np.eye(4)
    base_z = base_matrix[:3,2] 
    angle_0 = compute_angle_between_vectors(object_z, base_z)
    print("angle_0", angle_0)
    if abs(angle_0) < 30:
        print("object Stranding Up")
        if angle_0 > 0:
            print("Properly ") # 올바르게 세워져 있음
            defined_vector = np.array([0, 0, 0.083 + 0.077 + 0.018])  # 0.083 objcet offset +  
            defined_translation = pose[:3, 3] + pose[:3, :3] @ defined_vector
            transform[:3, 3] = defined_translation
        else:
            print("Not Properly") # 거꾸로 세워져있음
            defined_vector = np.array([0, 0, -(0.083 + 0.077 + 0.018)])  # 0.083 objcet offset +  
            defined_translation = pose[:3, 3] + pose[:3, :3] @ defined_vector
            transform[:3, 3] = defined_translation
            correction = np.eye(4)
            from scipy.spatial.transform import Rotation as R
            correction[:3, :3] = R.from_euler('x', 180, degrees=True).as_matrix()   
            transform = transform @ correction
    # elif abs(angle_0) > 70:
    #     print("Cant grasp")
        # transform = None
    else:
        print("object lying down") # object 누워있음
        object_y = base2target[:3,1]
        base_z = base_matrix[:3,2] 
        angle_1 = compute_angle_between_vectors(object_y, base_z)
        print(angle_1)
        from scipy.spatial.transform import Rotation as R
        if angle_1 > 0:
            print("Properly") # 똑바로 누워있음
            defined_vector = np.array([0, 0.04 + 0.077 + 0.018, 0])  # 0.083 objcet offset +  
            defined_translation = pose[:3, 3] + pose[:3, :3] @ defined_vector
            transform[:3, 3] = defined_translation
            correction = np.eye(4)
            correction[:3, :3] = R.from_euler('x', -90, degrees=True).as_matrix()   
            transform = transform @ correction
        else:
            print("Not Properly") # 거꾸로 누워있음
            defined_vector = np.array([0, 0.04 + 0.077 + 0.018, 0])  # 0.083 objcet offset +  
            defined_translation = pose[:3, 3] - pose[:3, :3] @ defined_vector
            transform[:3, 3] = defined_translation
            correction = np.eye(4)
            correction[:3, :3] = R.from_euler('x', 90, degrees=True).as_matrix()   
            transform = transform @ correction


    from scipy.spatial.transform import Rotation as R
    # 보정 회전: x축 기준 -90도 회전
    correction = np.eye(4)
    correction[:3, :3] = R.from_euler('y', 180, degrees=True).as_matrix()
    # 보정 적용
    transform = transform @ correction
    

    grasp_data = {
      "translation": np.array(transform[:3, 3]).tolist(),
      "rotation   ": np.array(transform[:3, :3]).tolist(),
      "transformation" : np.array(transform).tolist(),
    }

    file_path  = '/home/vision/packages/FoundationPose/result/best_grasp_panda.json'
    with open(file_path, "w") as json_file:
        json.dump(grasp_data, json_file, indent=4)
    print("Best grasp Panda properties saved as JSON.")


    # CAD Color 저장
    cad_colors_path = os.path.join(output_dir, "cad_colors.npy")
    np.save(cad_colors_path, cad_colors)
    # print(f"CAD Color 저장 완료: {cad_colors_path}")
    
    config_data = {
              "ARM" : ARM_ID,
              "mesh_path": args.mesh_file,
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
#   args.mesh_file = "/media/vision/data_4TB/grasp/models/models/ycb/004_sugar_box/textured_simple.obj"
#   args.mesh_file = "/media/vision/data_4TB/grasp/models/models/ycb/0/24_bowl/textured_simple.obj"
#   args.mesh_file = "/media/vision/data_4TB/grasp/models/models/ycb/trans_cup_1/trans_cup_mm.obj"
#   args.mesh_file = "/media/vision/data_4TB/grasp/models/models/ycb/006_mustard_bottle/textured_simple.obj"
#   args.mesh_file = "/media/vision/data_4TB/grasp/models/models/ycb/004_sugar_box/textured_simple.obj"
#   args.mesh_file = "/media/vision/data_4TB/grasp/graspnet/models/005/textured.obj" # banana
#   args.mesh_file = "/media/vision/data_4TB/grasp/graspnet/models/002/textured.obj" # tomato soup
  mesh = trimesh.load(args.mesh_file)
#   mesh.sample_points_uniformly(number_of_points=args.num_point)

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

  intrinsic = get_realsense_intrinsic()


  dict_intr = {'width': intrinsic.width,
               'height': intrinsic.height,
               'cx': intrinsic.ppx,
               'cy': intrinsic.ppy,
               'fx': intrinsic.fx,
               'fy': intrinsic.fy}
  with open('/home/vision/packages/FoundationPose/result/intr.yaml', 'w') as file:
      yaml.dump(dict_intr, file)

  # vis.run()
  # for i in range(len(reader.color_files)):
  i = 0
  while True:
    logging.info(f'i:{i}')
    # color = reader.get_color(i)
    # depth = reader.get_depth(i)

    color, depth = get_realsense_data()
    


    # depth = (depth * depth_scale * 1000).astype(np.float32)
    depth = (depth).astype(np.float32)


    H, W = color.shape[:2]
    color = cv2.resize(color, (W,H), interpolation=cv2.INTER_NEAREST)
    depth = cv2.resize(depth, (W,H), interpolation=cv2.INTER_NEAREST)
    # depth[(depth<0.1) | (depth>=np.inf)] = 0
    depth[(depth>=np.inf)] = 0

    
    # cam_K =  np.array([[612.6182250976562, 0, 318.95758056640625],
    #             [0, 612.7216796875, 240.05343627929688],
    #             [0, 0, 1]])
    
    # cam_K =  np.array([[intrinsic.fx, 0, intrinsic.ppx],
    #             [0, intrinsic.fy, intrinsic.ppy],
    #             [0, 0, 1]])
    
    # L515
    cam_K =  np.array([[607.01904296875, 0, 320.51519775390625],
                [0, 607.4635009765625, 236.34046936035156],
                [0, 0, 1]])
    

    print("\n\n cam_K : ", cam_K)
    if i==0:
      # mask = reader.get_mask(0).astype(bool)
      mask = get_mask_grounded(color, depth,args.text_prompt)
    #   mask = get_mask_click_and_crop(color , depth)
      
      # plt.figure(figsize=(8, 8))
      # plt.imshow(mask, cmap='gray')  # True는 흰색, False는 검은색으로 표시됨
      # plt.title("Mask Visualization")
      # plt.axis('off')  # 축 숨기기
      # plt.show()
      
      
      pose = est.register(K=cam_K, rgb=color, depth=depth, ob_mask=mask, iteration=args.est_refine_iter)

    else:
      pose = est.track_one(rgb=color, depth=depth, K=cam_K, iteration=args.track_refine_iter)

    i += 1
    break

    if i == args.track_refine_iter:
        break

    # os.makedirs(f'{debug_dir}/ob_in_cam', exist_ok=True)
    # np.savetxt(f'{debug_dir}/ob_in_cam/{reader.id_strs[i]}.txt', pose.reshape(4,4))

  if debug>=1:
    center_pose = pose@np.linalg.inv(to_origin)
    bbox_coord = pose@np.linalg.inv(to_origin)
    vis = draw_posed_3d_box(cam_K, img=color, ob_in_cam=center_pose, bbox=bbox)
    vis = draw_xyz_axis(color, ob_in_cam=center_pose, scale=0.1, K=cam_K, thickness=3, transparency=0, is_input_rgb=True)
  #   cv2.imshow('1', vis[...,::-1])
    cv2.imwrite(f'/home/vision/packages/FoundationPose/result/output_pose.png', vis[...,::-1])
  #   cv2.waitKey(1)  
    visualize_with_open3d(color, depth, cam_K, pose, bbox,bbox_coord,  mesh)
      # cv2.imshow('1', vis[...,::-1])
      # cv2.waitKey(1)
    

