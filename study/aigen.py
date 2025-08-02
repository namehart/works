import math
import numpy as np
from stl import mesh

# 定义数学常量
sqrt3 = math.sqrt(3)

# 定义点列表（14个点）
points_list = [
    (0, 0, 0),              # 0
    (2, 0, 0),              # 1
    (1, sqrt3, 0),          # 2
    (0, 0, 2),              # 3
    (2, 0, 2),              # 4
    (1, sqrt3, 2),          # 5
    (1, sqrt3/3, 0),        # 6
    (1, sqrt3/3, 2),        # 7
    (1, sqrt3/3, 1),        # 8
    (1.5, sqrt3/2, 0),      # 9
    (1.5, sqrt3/2, 2),      # 10
    (0.5, sqrt3/6, 2),      # 11
    (2/3, 0, 2),            # 12
    (1/3, sqrt3/3, 2)       # 13
]

# 定义边列表（16条边）
edges = [
    (points_list[0], points_list[3]),   # 0:(0,0,0)->(0,0,2)
    (points_list[1], points_list[4]),   # 1:(2,0,0)->(2,0,2)
    (points_list[2], points_list[5]),   # 2:(1,√3,0)->(1,√3,2)
    (points_list[0], points_list[1]),   # 3:(0,0,0)->(2,0,0)
    (points_list[1], points_list[2]),   # 4:(2,0,0)->(1,√3,0)
    (points_list[2], points_list[0]),   # 5:(1,√3,0)->(0,0,0)
    (points_list[3], points_list[4]),   # 6:(0,0,2)->(2,0,2)
    (points_list[4], points_list[5]),   # 7:(2,0,2)->(1,√3,2)
    (points_list[5], points_list[3]),   # 8:(1,√3,2)->(0,0,2)
    (points_list[6], points_list[7]),   # 9:(1,√3/3,0)->(1,√3/3,2)
    (points_list[9], points_list[10]),  # 10:(1.5,√3/2,0)->(1.5,√3/2,2)
    (points_list[9], points_list[8]),   # 11:(1.5,√3/2,0)->(1,√3/3,1)
    (points_list[8], points_list[11]),  # 12:(1,√3/3,1)->(0.5,√3/6,2)
    (points_list[12], points_list[13]), # 13:(2/3,0,2)->(1/3,√3/3,2)
    (points_list[1], points_list[12]),  # 14:(2,0,0)->(2/3,0,2)
    (points_list[2], points_list[13])   # 15:(1,√3,0)->(1/3,√3/3,2)
]

def create_cube(center, half_size=0.05):
    """为给定中心点创建立方体的三角形面"""
    x, y, z = center
    vertices = [
        (x - half_size, y - half_size, z - half_size),
        (x + half_size, y - half_size, z - half_size),
        (x + half_size, y + half_size, z - half_size),
        (x - half_size, y + half_size, z - half_size),
        (x - half_size, y - half_size, z + half_size),
        (x + half_size, y - half_size, z + half_size),
        (x + half_size, y + half_size, z + half_size),
        (x - half_size, y + half_size, z + half_size)
    ]
    # 定义立方体的三角形面（12个三角形）
    faces = [
        [0, 3, 1], [1, 3, 2],  # 底面
        [4, 5, 7], [5, 6, 7],  # 顶面
        [0, 1, 4], [1, 5, 4],  # 前面
        [1, 2, 5], [2, 6, 5],  # 右面
        [2, 3, 6], [3, 7, 6],  # 后面
        [3, 0, 7], [0, 4, 7]   # 左面
    ]
    return [(vertices[i], vertices[j], vertices[k]) for i, j, k in faces]

def create_beam(start, end, half_width=0.05):
    """创建连接两点的长方体梁的三角形面"""
    start = np.array(start)
    end = np.array(end)
    direction = end - start
    length = np.linalg.norm(direction)
    
    if length < 1e-8:  # 处理重合点
        return []
    
    z_axis = direction / length
    
    # 创建垂直向量
    if abs(z_axis[0]) > 1e-6 or abs(z_axis[2]) > 1e-6:
        up_vec = np.array([0, 1, 0])
    else:
        up_vec = np.array([1, 0, 0])
    
    x_axis = np.cross(up_vec, z_axis)
    if np.linalg.norm(x_axis) < 1e-8:
        x_axis = np.array([1, 0, 0])
        y_axis = np.cross(z_axis, x_axis)
    else:
        x_axis = x_axis / np.linalg.norm(x_axis)
        y_axis = np.cross(z_axis, x_axis)
    
    y_axis = y_axis / np.linalg.norm(y_axis)
    
    # 计算梁的8个顶点
    s1 = start + half_width * (x_axis + y_axis)
    s2 = start + half_width * (x_axis - y_axis)
    s3 = start + half_width * (-x_axis - y_axis)
    s4 = start + half_width * (-x_axis + y_axis)
    e1 = end + half_width * (x_axis + y_axis)
    e2 = end + half_width * (x_axis - y_axis)
    e3 = end + half_width * (-x_axis - y_axis)
    e4 = end + half_width * (-x_axis + y_axis)
    
    vertices = [s1, s2, s3, s4, e1, e2, e3, e4]
    
    # 定义长方体梁的三角形面（12个三角形）
    faces = [
        # 起点端
        [0, 1, 2], [0, 2, 3],
        # 终点端
        [4, 6, 5], [4, 7, 6],
        # 侧面
        [0, 4, 1], [1, 4, 5],
        [1, 5, 2], [2, 5, 6],
        [2, 6, 3], [3, 6, 7],
        [3, 7, 0], [0, 7, 4]
    ]
    return [(vertices[i], vertices[j], vertices[k]) for i, j, k in faces]

# 收集所有三角形面
all_faces = []

# 为每个点创建立方体
for point in points_list:
    all_faces.extend(create_cube(point))

# 为每条边创建梁
for edge in edges:
    all_faces.extend(create_beam(edge[0], edge[1]))

# 创建STL网格
stl_mesh = mesh.Mesh(np.zeros(len(all_faces), dtype=mesh.Mesh.dtype))
for i, face in enumerate(all_faces):
    for j in range(3):
        stl_mesh.vectors[i][j] = face[j]

# 保存STL文件
stl_mesh.save('frame_model.stl')
print("STL文件已保存为 'frame_model.stl'")