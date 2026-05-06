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

# 필요한 파일 import
import grounded_sam_demo as gs  # 예: some_file.py에서 정의된 함수나 클래스 사용

# sys.path.append(os.path.join("/home/vision/packages/Grounded-Segment-Anything", "GroundingDINO"))
# sys.path.append(os.path.join("/home/vision/packages/Grounded-Segment-Anything", "segment_anything"))


# Grounding DINO
import GroundingDINO.groundingdino.datasets.transforms as T
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap



# segment anything
from segment_anything import (
    sam_model_registry,
    sam_hq_model_registry,
    SamPredictor
)




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


depth_sensor.set_option(rs.option.depth_units, 0.001)
unit_value = depth_sensor.get_option(rs.option.depth_units) # 기본 0.001
print(unit_value)
depth_scale = depth_sensor.get_depth_scale()

user_input = None
def get_realsense_data():
  frames = pipeline.wait_for_frames()
  aligned_frames = align.process(frames)
  color_frame = aligned_frames.get_color_frame()
  aligned_depth_frame = aligned_frames.get_depth_frame()

  color = np.asanyarray(color_frame.get_data())
  depth = np.asanyarray(aligned_depth_frame.get_data())/1e3
  color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)

  return color, depth
  
def get_mask(color, depth):
  # cfg
  config_file = "/home/vision/packages/Grounded-Segment-Anything/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"  # change the path of the model config file
  grounded_checkpoint = "/home/vision/packages/Grounded-Segment-Anything/groundingdino_swint_ogc.pth"  # change the path of the model
  sam_version = "vit_h"
  sam_checkpoint = "/home/vision/packages/Grounded-Segment-Anything/sam_vit_h_4b8939.pth"
  text_prompt = "the object"
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
  transformed_boxes = predictor.transform.apply_boxes_torch(boxes_filt, color.shape[:2]).to(device)
  masks, _, _ = predictor.predict_torch(
      point_coords = None,
      point_labels = None,
      boxes = transformed_boxes.to(device),
      multimask_output = False,
  )

  selected_mask = None
  logging.info(f'mask shape:{masks.shape}')


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
    plt.show()


    if user_input == 'y':  # User selects this mask
      selected_mask = mask_np
      break
    elif user_input == 'n':
      logging.info(f'NONONO!!!')
      continue
    
    # key = cv2.waitKey(0)  # Wait for key press
    # if key == ord('y'):  # User selects this mask
    #   selected_mask = mask.cpu().numpy()
    #   break
    # elif key == ord('n'):
    #   continue

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

def visualize_with_open3d(color, depth, cam_K, pose, bbox, mesh):
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
    transformed_corners = (pose[:3, :3] @ corners.T).T + pose[:3, 3]

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
     # 3. CAD 모델 축 조정
    # cad_axis_adjustment = np.array([
    #     [0,  0,  1,  0],  # Z → X
    #     [0,  1,  0,  0],  # Y 동일
    #     [1,  0,  0,  0],  # X → Z
    #     [0,  0,  0,  1],  # 동차 좌표
    # ])
    # adjusted_pose = pose @ cad_axis_adjustment
    # Z축 반전 행렬
    z_flip = np.eye(4)
    z_flip[2, 2] = -1
    adjusted_pose = pose @ z_flip
    z_90 = get_z_rotation_matrix_minus_90()
    x_neg_90 = np.array([
        [1,  0,  0,  0],
        [0,  0, -1,  0],  # Y → -Z
        [0,  1,  0,  0],  # Z → Y
        [0,  0,  0,  1]
    ])
    adjusted_pose = adjusted_pose @ x_neg_90
    cvcam_in_glcam = np.array([[1,0,0,0],
                          [0,-1,0,0],
                          [0,0,-1,0],
                          [0,0,0,1]])
    pose = pose @ cvcam_in_glcam
    pose = pose @ z_90
    # pose = pose @ x_neg_90
    mesh_copy.apply_transform(pose)  # Trimesh 객체 변환

    # o3d_mesh = o3d.geometry.TriangleMesh()
    # o3d_mesh.vertices = o3d.utility.Vector3dVector(mesh_copy.vertices)
    # o3d_mesh.triangles = o3d.utility.Vector3iVector(mesh_copy.faces)

    # if hasattr(mesh_copy.visual, 'vertex_colors') and mesh_copy.visual.vertex_colors is not None:
    #     o3d_mesh.vertex_colors = o3d.utility.Vector3dVector(mesh_copy.visual.vertex_colors[:, :3] / 255.0)


    cad_points = np.asarray(mesh_copy.vertices)  # CAD 파일의 vertices 가져오기
    cad_colors = (
        np.asarray(mesh_copy.visual.vertex_colors[:, :3]) / 255.0
        if hasattr(mesh_copy.visual, 'vertex_colors') and mesh_copy.visual.vertex_colors is not None
        else np.full_like(cad_points, [0.5, 0.5, 0.5])  # 기본 회색
    )

    cad_cloud = o3d.geometry.PointCloud()
    cad_cloud.points = o3d.utility.Vector3dVector(cad_points)
    cad_cloud.colors = o3d.utility.Vector3dVector(cad_colors)
    
    
    # 시각화
    geometries = [cloud, coordinate_frame, cad_cloud, line_set]
    # if oriented_box:
        # geometries.append(oriented_box)

    o3d.visualization.draw_geometries(geometries)
    

    #####################################3
    cad_points = np.asarray(mesh_copy.vertices)
    cad_colors = (
        np.asarray(mesh_copy.visual.vertex_colors[:, :3]) / 255.0
        if hasattr(mesh_copy.visual, 'vertex_colors') and mesh_copy.visual.vertex_colors is not None
        else np.full_like(cad_points, [0.5, 0.5, 0.5])  # 기본 회색
    )
    # CAD Point Cloud 저장
    output_dir = '/home/vision/packages/FoundationPose'
    cad_points_path = os.path.join(output_dir, "cad_pointcloud.npy")
    np.save(cad_points_path, cad_points)
    print(f"CAD Point Cloud 저장 완료: {cad_points_path}")

    # CAD Color 저장
    cad_colors_path = os.path.join(output_dir, "cad_colors.npy")
    np.save(cad_colors_path, cad_colors)
    print(f"CAD Color 저장 완료: {cad_colors_path}")
    
    
        # 엔터키를 눌러 다음 프레임으로 이동
    print("Press Enter to move to the next frame...")
    while True:
        cv2.imshow(color)
        key = cv2.waitKey(1) & 0xFF
        if key == 13:  # ESC 키
            print("ESC 키 눌림. i를 0으로 초기화합니다.")
            o3d.visualization.destory_window()
            return
            break
        
    return 
    o3d.visualization.destory_window()



if __name__=='__main__':
  parser = argparse.ArgumentParser()
  code_dir = os.path.dirname(os.path.realpath(__file__))
  parser.add_argument('--mesh_file', type=str, default=f'{code_dir}/demo_data/mustard0/mesh/untitled.obj')
  parser.add_argument('--test_scene_dir', type=str, default=f'{code_dir}/demo_data/mustard0')
  parser.add_argument('--est_refine_iter', type=int, default=5)
  parser.add_argument('--track_refine_iter', type=int, default=2)
  parser.add_argument('--debug', type=int, default=1)
  
  parser.add_argument('--debug_dir', type=str, default=f'{code_dir}/debug')
  args = parser.parse_args()

  set_logging_format()
  set_seed(0)
#   args.mesh_file = "/media/vision/data_4TB/grasp/models/models/ycb/024_bowl/textured_simple.obj"
#   args.mesh_file = "/media/vision/data_4TB/grasp/models/models/ycb/004_sugar_box/textured_simple.obj"
  mesh = trimesh.load(args.mesh_file)

  debug = args.debug
  debug_dir = args.debug_dir
  os.system(f'rm -rf {debug_dir}/* && mkdir -p {debug_dir}/track_vis {debug_dir}/ob_in_cam')

  to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
  bbox = np.stack([-extents/2, extents/2], axis=0).reshape(2,3)

  # sungjun edit = normal vector와 vertice 시각화
  # Convert trimesh to Open3D mesh
  o3d_mesh = o3d.geometry.TriangleMesh()
  o3d_mesh.vertices = o3d.utility.Vector3dVector(mesh.vertices)
  o3d_mesh.triangles = o3d.utility.Vector3iVector(mesh.faces)
  o3d_mesh.compute_vertex_normals()
  
  # Create PointCloud for vertices
  point_cloud = o3d.geometry.PointCloud()
  point_cloud.points = o3d.utility.Vector3dVector(mesh.vertices)
  point_cloud.normals = o3d.utility.Vector3dVector(np.asarray(mesh.vertex_normals).copy())  # Fix here
  
  # Create LineSet to visualize normals
  line_set = o3d.geometry.LineSet()
  scale = 0.01  # Adjust the length of the normals
  vertices = np.asarray(mesh.vertices)
  normals = np.asarray(mesh.vertex_normals)
  
  # Compute start and end points for normals
  start_points = vertices
  end_points = vertices + scale * normals
  all_points = np.vstack([start_points, end_points])  # Combine start and end points
  
  # Define lines connecting start and end points
  lines = [[i, i + len(vertices)] for i in range(len(vertices))]
  line_set.points = o3d.utility.Vector3dVector(all_points)
  line_set.lines = o3d.utility.Vector2iVector(lines)
  
  # Set normal vector colors (red)
  line_set.colors = o3d.utility.Vector3dVector([[1, 0, 0] for _ in range(len(lines))])
  
  # Visualize mesh, points, and normals
  o3d.visualization.draw_geometries([o3d_mesh, point_cloud, line_set],
                                    window_name="Mesh Vertices and Normals",
                                    width=800, height=600,
                                    left=50, top=50,
                                    mesh_show_back_face=True)
    # sungjun edit end

  scorer = ScorePredictor()
  refiner = PoseRefinePredictor()
  glctx = dr.RasterizeCudaContext()
  est = FoundationPose(model_pts=mesh.vertices, model_normals=mesh.vertex_normals, mesh=mesh, scorer=scorer, refiner=refiner, debug_dir=debug_dir, debug=debug, glctx=glctx)
  logging.info("estimator initialization done")



  # vis.run()
  # for i in range(len(reader.color_files)):
  i = 0
  while True:
    logging.info(f'i:{i}')
    # color = reader.get_color(i)
    # depth = reader.get_depth(i)
    color, depth = get_realsense_data()
    depth = (depth * depth_scale * 1000).astype(np.float32)

    if depth.dtype != np.float32:
        depth = depth.astype(np.float32)  # Convert to float32

    H, W = color.shape[:2]
    color = cv2.resize(color, (W,H), interpolation=cv2.INTER_NEAREST)
    depth = cv2.resize(depth, (W,H), interpolation=cv2.INTER_NEAREST)
    depth[(depth<0.1) | (depth>=np.inf)] = 0
    cam_K =  np.array([[612.6182250976562, 0, 318.95758056640625],
                [0, 612.7216796875, 240.05343627929688],
                [0, 0, 1]])
    if i==0:
      # mask = reader.get_mask(0).astype(bool)
      mask = get_mask(color, depth)

      
      # plt.figure(figsize=(8, 8))
      # plt.imshow(mask, cmap='gray')  # True는 흰색, False는 검은색으로 표시됨
      # plt.title("Mask Visualization")
      # plt.axis('off')  # 축 숨기기
      # plt.show()
      
      
      pose = est.register(K=cam_K, rgb=color, depth=depth, ob_mask=mask, iteration=args.est_refine_iter)

      if debug>=3:
        m = mesh.copy()
        m.apply_transform(pose)
        m.export(f'{debug_dir}/model_tf.obj')
        xyz_map = depth2xyzmap(depth, reader.K)
        valid = depth>=0.001
        pcd = toOpen3dCloud(xyz_map[valid], color[valid])
        o3d.io.write_point_cloud(f'{debug_dir}/scene_complete.ply', pcd)
    else:
      pose = est.track_one(rgb=color, depth=depth, K=cam_K, iteration=args.track_refine_iter)

    # os.makedirs(f'{debug_dir}/ob_in_cam', exist_ok=True)
    # np.savetxt(f'{debug_dir}/ob_in_cam/{reader.id_strs[i]}.txt', pose.reshape(4,4))

    if debug>=1:
      center_pose = pose@np.linalg.inv(to_origin)
      vis = draw_posed_3d_box(cam_K, img=color, ob_in_cam=center_pose, bbox=bbox)
      vis = draw_xyz_axis(color, ob_in_cam=center_pose, scale=0.1, K=cam_K, thickness=3, transparency=0, is_input_rgb=True)
    #   visualize_with_open3d(color, depth, cam_K, center_pose, bbox, mesh)

      cv2.imshow('1', vis[...,::-1])
      cv2.waitKey(1)


    if debug>=2:
      os.makedirs(f'{debug_dir}/track_vis', exist_ok=True)
      imageio.imwrite(f'{debug_dir}/track_vis/{reader.id_strs[i]}.png', vis)
    
    i += 1

