import os
import trimesh
import open3d as o3d
from obj_params import configs as cfgs 
import matplotlib.pyplot as plt

def simplify_mesh_trimesh(input_path, output_path, target_ratio=0.3, model_name = None):
    # Trimesh로 불러오기
    mesh = trimesh.load(input_path)
    


    index_fill = (str(int(model_name)).zfill(3))
    sample_voxel_size, model_voxel_size, model_num_sample = cfgs[index_fill]

    # 간소화
    ratio = (1-sample_voxel_size*100)*0.075
    target_faces = int((mesh.faces.shape[0]) * (ratio))
    print("ratioo ", ratio)
    simplified = mesh.simplify_quadric_decimation(target_faces)
    print("Before : ", (mesh.vertices.shape[0]))
    print("After  : ", (simplified.vertices.shape[0]))
    # 저장
    # o3d.io.write_triangle_mesh(output_path, simplified)
    print(f"Simplified mesh saved to: {output_path}")

def get_vertex_counts_from_meshes(folder_path):
    vertex_counts = []
    for filename in os.listdir(folder_path):
        if filename == "textured.obj":
            mesh_path = os.path.join(folder_path, filename)
            mesh = trimesh.load(mesh_path, force='mesh')
            if mesh.is_empty:
                continue
            vertex_counts.append(len(mesh.vertices))
    return vertex_counts

def get_vertex_counts_from_simple_meshes(folder_path):
    vertex_counts = []
    for filename in os.listdir(folder_path):
        if filename == "textured_simple.obj":
            mesh_path = os.path.join(folder_path, filename)
            mesh = trimesh.load(mesh_path, force='mesh')
            if mesh.is_empty:
                continue
            vertex_counts.append(len(mesh.vertices))
    return vertex_counts



if __name__=='__main__':
    # 전체 폴더 루프
    # 일반 mesh 폴더 경로
  
  # 누적 리스트
  original_vertex_counts = []
  simplified_vertex_counts = []

  # root_dir = '/media/vision/data_4TB/pose_estimation/instance/YCB_Video_Models/models'
  root_dir = '/media/vision/data_4TB/grasp/graspnet/models'
  for folder_name in sorted(os.listdir(root_dir)):
    folder_path = os.path.join(root_dir, folder_name)
    print(f"{folder_path} 처리중 ...")
    if not os.path.isdir(folder_path):
        continue
    
    # vertex 개수 가져오기
    original = get_vertex_counts_from_meshes(folder_path)
    simplified = get_vertex_counts_from_simple_meshes(folder_path)

    original_vertex_counts.extend(original)
    simplified_vertex_counts.extend(simplified)
    

import numpy as np

# mesh index 기준으로 x축 구성
x = np.arange(len(original_vertex_counts))

plt.figure(figsize=(14, 6))
plt.bar(x - 0.2, original_vertex_counts, width=0.4, label='Original', color='skyblue')
plt.bar(x + 0.2, simplified_vertex_counts, width=0.4, label='Simplified', color='salmon')
plt.xlabel('Mesh Index')
plt.ylabel('Number of Vertices')
plt.title('Vertex Counts per Mesh (Original vs Simplified)')
plt.xticks(x)  # 필요 없으면 생략 가능
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()


  

  

    