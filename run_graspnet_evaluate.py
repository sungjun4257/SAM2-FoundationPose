# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

# graspnet dataset의 pose 가 foundation pose와 비슷한지
from estimater import *
from datareader import *
import argparse

class CameraInfo():
  ''' Author: chenxi-wang
  Camera intrinsics for point cloud generation.
  '''
  def __init__(self, width, height, fx, fy, cx, cy, scale):
    self.width = width
    self.height = height
    self.fx = fx
    self.fy = fy
    self.cx = cx
    self.cy = cy
    self.scale = scale

def create_point_cloud_from_depth_image(depth, camera, organized=True):
  assert(depth.shape[0] == camera.height and depth.shape[1] == camera.width)
  xmap = np.arange(camera.width)
  ymap = np.arange(camera.height)
  xmap, ymap = np.meshgrid(xmap, ymap)
  points_z = depth / camera.scale
  points_x = (xmap - camera.cx) * points_z / camera.fx
  points_y = (ymap - camera.cy) * points_z / camera.fy
  cloud = np.stack([points_x, points_y, points_z], axis=-1)
  if not organized:
    cloud = cloud.reshape([-1, 3])
  return cloud

def transform_point_cloud(cloud, transform, format='4x4'):
  """ Transform points to new coordinates with transformation matrix.
      Input:
          cloud: [np.ndarray, (N,3), np.float32]
              points in original coordinates
          transform: [np.ndarray, (3,3)/(3,4)/(4,4), np.float32]
              transformation matrix, could be rotation only or rotation+translation
          format: [string, '3x3'/'3x4'/'4x4']
              the shape of transformation matrix
              '3x3' --> rotation matrix
              '3x4'/'4x4' --> rotation matrix + translation matrix
      Output:
          cloud_transformed: [np.ndarray, (N,3), np.float32]
              points in new coordinates
  """
  if not (format == '3x3' or format == '4x4' or format == '3x4'):
    raise ValueError('Unknown transformation format, only support \'3x3\' or \'4x4\' or \'3x4\'.')
  if format == '3x3':
    cloud_transformed = np.dot(transform, cloud.T).T
  elif format == '4x4' or format == '3x4':
    ones = np.ones(cloud.shape[0])[:, np.newaxis]
    cloud_ = np.concatenate([cloud, ones], axis=1)
    cloud_transformed = np.dot(transform, cloud_.T).T
    cloud_transformed = cloud_transformed[:, :3]
  return cloud_transformed

def get_workspace_mask(cloud, seg, trans=None, organized=True, outlier=0):
  """ Keep points in workspace as input.
      Input:
          cloud: [np.ndarray, (H,W,3), np.float32]
              scene point cloud
          seg: [np.ndarray, (H,W,), np.uint8]
              segmantation label of scene points
          trans: [np.ndarray, (4,4), np.float32]
              transformation matrix for scene points, default: None.
          organized: [bool]
              whether to keep the cloud in image shape (H,W,3)
          outlier: [float]
              if the distance between a point and workspace is greater than outlier, the point will be removed
              
      Output:
          workspace_mask: [np.ndarray, (H,W)/(H*W,), np.bool]
              mask to indicate whether scene points are in workspace
  """
  if organized:
    h, w, _ = cloud.shape
    cloud = cloud.reshape([h*w, 3])
    seg = seg.reshape(h*w)
  if trans is not None:
    cloud = transform_point_cloud(cloud, trans)
  foreground = cloud[seg>0]
  xmin, ymin, zmin = foreground.min(axis=0)
  xmax, ymax, zmax = foreground.max(axis=0)
  mask_x = ((cloud[:,0] > xmin-outlier) & (cloud[:,0] < xmax+outlier))
  mask_y = ((cloud[:,1] > ymin-outlier) & (cloud[:,1] < ymax+outlier))
  mask_z = ((cloud[:,2] > zmin-outlier) & (cloud[:,2] < zmax+outlier))
  workspace_mask = (mask_x & mask_y & mask_z)
  if organized:
    workspace_mask = workspace_mask.reshape([h, w])
  return workspace_mask

if __name__=='__main__':
  parser = argparse.ArgumentParser()
  code_dir = os.path.dirname(os.path.realpath(__file__))
  parser.add_argument('--mesh_file', type=str, default=f'{code_dir}/demo_data/mustard0/mesh/textured_simple.obj')
  # parser.add_argument('--test_scene_dir', type=str, default=f'{code_dir}/demo_data/mustard0')
  parser.add_argument('--test_scene_dir', type=str, default='/media/vision/data_4TB/grasp/graspnet/scenes/scene_0000')
  parser.add_argument('--est_refine_iter', type=int, default=5)
  parser.add_argument('--track_refine_iter', type=int, default=2)
  parser.add_argument('--debug', type=int, default=1)
  parser.add_argument('--debug_dir', type=str, default=f'{code_dir}/debug')
  args = parser.parse_args()

  set_logging_format()
  set_seed(0)

  mesh = trimesh.load(args.mesh_file)

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

  # reader = YcbineoatReader(video_dir=args.test_scene_dir, shorter_side=None, zfar=np.inf)
  reader = GraspnetReader(video_dir=args.test_scene_dir,downscale=1, shorter_side=None, zfar=np.inf)

  # for scene_number in range(199):
  #   # scene_number = 0
  #   dir = f"/media/vision/data_4TB/grasp/graspnet/scenes/scene_{scene_number:04d}"
  #   reader = GraspnetReader(video_dir=dir,downscale=1, shorter_side=None, zfar=np.inf)

  #   obj_indexs = reader.get_instance_ids_in_image(scene_number)

    
  #   for i in range(len(reader.color_files)):
  #     poses_list = []
  #     color = reader.get_color(i)
  #     depth = reader.get_depth(i)
  #     # seg = reader.get_mask(i)
  #     meta = reader.get_meta(i)

  #     intrinsic = meta['intrinsic_matrix']
  #     factor_depth = meta['factor_depth']

  #     # camera = CameraInfo(1280.0, 720.0, intrinsic[0][0], intrinsic[1][1], intrinsic[0][2], intrinsic[1][2], factor_depth)
  #     # cloud_real = create_point_cloud_from_depth_image(depth, camera, organized=True)
  #     # depth_mask = (depth > 0)
  #     # root = args.test_scene_dir
  #     # camera_poses = np.load(os.path.join(root, 'scenes', scene_number, 'realsense', 'camera_poses.npy'))
  #     # align_mat = np.load(os.path.join(root, 'scenes', scene_number, 'realsense', 'cam0_wrt_table.npy'))
  #     # trans = np.dot(align_mat, camera_poses[i])
  #     # workspace_mask = get_workspace_mask(cloud_real, seg, trans=trans, organized=True, outlier=0.02)
  #     # mask = (depth_mask & workspace_mask)
      
  #     # color = color[mask]
  #     # depth = depth[mask]
  #     # seg = seg[mask]

  #     # plt.imshow(depth, cmap='plasma')  # Depth 이미지에 색상 맵을 적용
  #     # plt.colorbar()  # 색상 바(Colorbar)를 추가하여 깊이 값에 대한 색상을 표시
  #     # plt.axis('off')  # 축 숨기기
  #     # plt.show()
  #     for j in range(len(obj_indexs)):
  #       logging.info(f'{scene_number}th scene - {i}th image - {j}th object')
  #       errs = []
  #       mesh = reader.get_gt_mesh(obj_indexs[j])
  #       # target_faces = int(mesh.faces.shape[0] * 0.1)
  #       # mesh = mesh.simplify_quadratic_decimation(target_faces)

  #       to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
  #       bbox = np.stack([-extents/2, extents/2], axis=0).reshape(2,3)

  #       est = FoundationPose(model_pts=mesh.vertices, model_normals=mesh.vertex_normals, mesh=mesh, scorer=scorer, refiner=refiner, debug_dir=debug_dir, debug=debug, glctx=glctx, model_num=obj_indexs[j])
  #       logging.info("estimator initialization done")
    
  #       mask = reader.get_mask(i, obj_indexs[j]).astype(bool)
  #       # logging.info(f'mask:{mask.shape}')
  #       # logging.info(f'mask:{mask.dtype}')
  #       # mask 시각화
  #       # plt.figure(figsize=(8, 8))
  #       # plt.imshow(mask, cmap='gray')  # True는 흰색, False는 검은색으로 표시됨
  #       # plt.title("Mask Visualization")
  #       # plt.axis('off')  # 축 숨기기
  #       # plt.show()
  #       # plt.figure(figsize=(8, 8))
  #       # plt.imshow(color)  # True는 흰색, False는 검은색으로 표시됨
  #       # plt.title("Mask Visualization")
  #       # plt.axis('off')  # 축 숨기기
  #       # plt.show()


  #       pose = est.register(K=reader.K, rgb=color, depth=depth, ob_mask=mask, iteration=args.est_refine_iter)
  #       gt = reader.get_gt_pose(i, j)
  #       err = est.add_err(pose.reshape(4,4) ,gt ,mesh.vertices)
  #       # errs.append(err)
  #       # auc = est.compute_auc_sklearn(errs)
  #       print("err ",err)
  #       # print("auc : ", auc)

        
  #       poses_list.append(pose.reshape(4, 4))

  #       # # 메모리 관련
  #       # torch.cuda.empty_cache()
  #       # allocated_memory = torch.cuda.memory_allocated()
  #       # print(f"Allocated memory: {allocated_memory / 1024 ** 2:.2f} MB")

  #       if debug>=3:
  #         m = mesh.copy()
  #         m.apply_transform(pose)
  #         m.export(f'{debug_dir}/model_tf.obj')
  #         xyz_map = depth2xyzmap(depth, reader.K)
  #         valid = depth>=0.001
  #         pcd = toOpen3dCloud(xyz_map[valid], color[valid])
  #         o3d.io.write_point_cloud(f'{debug_dir}/scene_complete.ply', pcd)
  #       # else:
  #       #   pose = est.track_one(rgb=color, depth=depth, K=reader.K, iteration=args.track_refine_iter)
  
  #       os.makedirs(f'{debug_dir}/ob_in_cam', exist_ok=True)
  #       np.savetxt(f'{debug_dir}/ob_in_cam/{reader.id_strs[i]}.txt', pose.reshape(4,4))

  #       # if debug>=1:
  #       #   center_pose = pose@np.linalg.inv(to_origin)
  #       #   vis = draw_posed_3d_box(reader.K, img=color, ob_in_cam=center_pose, bbox=bbox)
  #       #   vis = draw_xyz_axis(color, ob_in_cam=center_pose, scale=0.1, K=reader.K, thickness=3, transparency=0, is_input_rgb=True)
  #       #   cv2.imshow('1', vis[...,::-1])
  #       #   cv2.waitKey(1)


  #       if debug>=2:
  #         os.makedirs(f'{debug_dir}/track_vis', exist_ok=True)
  #         imageio.imwrite(f'{debug_dir}/track_vis/{reader.id_strs[i]}.png', vis)
      
  #     if poses_list:
  #       poses = np.stack(poses_list, axis=2)  # shape: (4, 4, N_objects)

  #       meta = {
  #           'poses': poses,
  #       }
  #       save_dir = f"/home/vision/packages/FoundationPose/graspnet_data_result/scene_{scene_number:04d}"
  #       os.makedirs(save_dir, exist_ok=True)  # 경로가 없으면 생성
        
  #       save_path = os.path.join(save_dir, f'{i:04d}.mat')
  #       scio.savemat(save_path, meta)
  #       print(f"✅ Saved: {save_path}")

  object_pose_add_lists = [[] for _ in range(88)]
  object_pose_auc_lists = [[] for _ in range(88)]

  # Object 단위로 저장
  for scene_number in range(189):
    # scene_number = 0
    dir = f"/media/vision/data_4TB/grasp/graspnet/scenes/scene_{scene_number:04d}"
    reader = GraspnetReader(video_dir=dir,downscale=1, shorter_side=None, zfar=np.inf)

    obj_indexs = reader.get_instance_ids_in_image(scene_number)

    poses_per_image = {i: {} for i in range(len(reader.color_files))}

    for i in range(len(reader.color_files)):

      for j in range(len(obj_indexs)):
        poses_list = []
        obj_id = obj_indexs[j]

        mesh = reader.get_gt_mesh(obj_indexs[j])

        to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
        bbox = np.stack([-extents/2, extents/2], axis=0).reshape(2,3)
      
        logging.info('\n--------------------------------------')
        logging.info(f'{scene_number}th scene - {i}th image - {j}th object')
        
        color = reader.get_color(i)
        depth = reader.get_depth(i)
        meta = reader.get_meta(i)

        intrinsic = meta['intrinsic_matrix']
        factor_depth = meta['factor_depth']

        mask = reader.get_mask(i, obj_indexs[j]).astype(bool)

        gt_pose = reader.get_gt_pose(i, j)
        predict_pose = reader.get_predict_pose(i,j)
        if predict_pose is None:
          print("predict_pose is None")
          continue
        if np.isnan(predict_pose).any():
          print("predict_pose contains NaN values")
          continue
        err = est.adds_err(predict_pose ,gt_pose ,mesh.vertices)
        object_pose_add_lists[obj_indexs[j]-1].append(err)
        auc = est.compute_auc_sklearn(object_pose_add_lists[obj_indexs[j]-1])
        object_pose_auc_lists[obj_indexs[j]-1].append(auc)
        print("\n object : ",obj_indexs[j]-1)
        print("err ",err)
        print("auc ",auc)




        # else:
        #   pose = est.track_one(rgb=color, depth=depth, K=reader.K, iteration=args.track_refine_iter)
  

        # if debug>=1:
        #   center_pose = pose@np.linalg.inv(to_origin)
        #   vis = draw_posed_3d_box(reader.K, img=color, ob_in_cam=center_pose, bbox=bbox)
        #   vis = draw_xyz_axis(color, ob_in_cam=center_pose, scale=0.1, K=reader.K, thickness=3, transparency=0, is_input_rgb=True)
        #   cv2.imshow('1', vis[...,::-1])
        #   cv2.waitKey(1)


        if debug>=2:
          os.makedirs(f'{debug_dir}/track_vis', exist_ok=True)
          imageio.imwrite(f'{debug_dir}/track_vis/{reader.id_strs[i]}.png', vis)
      
    
    import json
    object_data = {}
    
    for obj_id in range(88):
        object_data[obj_id] = {
            "ADD_Errors": object_pose_add_lists[obj_id],
            "AUC_Values": object_pose_auc_lists[obj_id]
        }
    
    with open("/home/vision/packages/FoundationPose/graspnet_data_result/object_wo_small_adds_auc_values.json", "w") as f:
        json.dump(object_data, f, indent=2)

    # for i, obj_pose_dict in poses_per_image.items():
    #   if obj_pose_dict:
    #     # 1. object pose들을 리스트로 모아서
    #     poses_list = list(obj_pose_dict.values())  # [(4,4), (4,4), ...]
    #     poses = np.stack(poses_list, axis=2)       # (4,4,N_objects)

    #     # 2. meta 포맷에 맞게 저장
    #     meta = {
    #         'poses': poses
    #     }

    #     save_dir = f"/home/vision/packages/FoundationPose/graspnet_data_result/scene_{scene_number:04d}"
    #     os.makedirs(save_dir, exist_ok=True)  # 경로가 없으면 생성
    #     save_path = os.path.join(save_dir, f'{i:04d}.mat')
    #     scio.savemat(save_path, meta)
    #     print(f"✅ Saved: {save_path}")
