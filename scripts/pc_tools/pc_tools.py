from transformations import euler_matrix
import numpy as np
import open3d as o3d

def transform_points(points, xyz, rpy):
    assert points.shape[-1] == 4 # xyzi format
    x, y, z = xyz
    roll, pitch, yaw = rpy
    pos, intensity = points[:,:3], points[:,3]
    M = euler_matrix(roll, pitch, yaw).astype(np.float32)
    M[0:3,3] = np.array([x, y, z])
    pos = np.hstack((pos, np.ones((pos.shape[0],1), dtype=np.float32)))
    pos = (M @ pos.T).T
    points = np.hstack((pos[:,:3], np.expand_dims(intensity,1)))
    return points

def downsample_points(points, voxel_size):  
    pos, intensity = points[:,:3], points[:,3]    
    pcd = o3d.t.geometry.PointCloud()
    pcd.point.positions = o3d.core.Tensor.from_numpy(pos)
    pcd.point.intensities = o3d.core.Tensor.from_numpy(intensity.reshape((-1,1)))
    pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
    pos = pcd.point.positions.numpy()
    intensity = pcd.point.intensities.numpy()
    points = np.hstack((pos, intensity))
    return points

def crop_points(points, min_bound, max_bound):  
    pos, intensity = points[:,:3], points[:,3]    
    pcd = o3d.t.geometry.PointCloud()
    pcd.point.positions = o3d.core.Tensor.from_numpy(pos)
    pcd.point.intensities = o3d.core.Tensor.from_numpy(intensity.reshape((-1,1)))
    cropping_box = o3d.t.geometry.AxisAlignedBoundingBox(
                min_bound=min_bound,
                max_bound=max_bound
            )
    pcd = pcd.crop(cropping_box)
    pos = pcd.point.positions.numpy()
    intensity = pcd.point.intensities.numpy()
    points = np.hstack((pos, intensity))
    return points
    