from sensor_msgs.msg import PointCloud2, PointField
import numpy as np
from std_msgs.msg import Header
from rclpy.clock import Clock

def numpy_to_rosmsg(points, parent_frame, stamp=None):
    d = points.shape[1]
    ros_dtype = PointField.FLOAT32
    dtype = np.float32
    itemsize = np.dtype(dtype).itemsize

    data = points.astype(dtype).tobytes()

    fields = [PointField(
        name=n, offset=i*itemsize, datatype=ros_dtype, count=1)
        for i, n in enumerate('xyzi')]
    fields[-1].name = 'intensity'
    
    if stamp is None:
        stamp = Clock().now()
    header = Header(frame_id=parent_frame, stamp=stamp.to_msg())

    return PointCloud2(
        header=header,
        height=1,
        width=points.shape[0],
        is_dense=True,
        is_bigendian=False,
        fields=fields,
        point_step=(itemsize * d),
        row_step=(itemsize * d * points.shape[0]),
        data=data
    )

def rosmsg_to_numpy(msg, lidar='velo'):
    points = np.zeros((msg.height*msg.width, 4), dtype=np.float32)
    if lidar=='velo':
        np_dtype = [('x', '<f4'), ('y','<f4'), ('z', '<f4'), ('i', '<f4')]
        np_pc = np.frombuffer(msg.data, dtype=np_dtype)
        points[:,0]=np_pc['x']
        points[:,1]=np_pc['y']
        points[:,2]=np_pc['z']
        points[:,3]=np_pc['i']
    elif lidar=='hesai':
        np_dtype = [('x', '<f4'), ('y', '<f4'), ('z', '<f4'), ('b0', '<f4'), ('i', '<f4'), ('b1', '<f4'), ('b2', '<f4'), ('b3', '<f4')]
        np_pc = np.frombuffer(msg.data, dtype=np_dtype)
        points[:,0]=np_pc['x']
        points[:,1]=np_pc['y']
        points[:,2]=np_pc['z']
        points[:,3]=np_pc['i']
    elif lidar=='hesai_bag':
        np_dtype = [('x', '<f4'), ('y', '<f4'), ('z', '<f4'), ('b0', '<f4'), ('i', '<u1'), ('b10', '<u1'), ('b11', '<u1'), ('b12', '<u1'), 
                    ('b2', '<f4'), ('b3', '<f4'), ('b4', '<f4'), ('b5', '<f4'), ('b6', '<f4'), ('b7', '<f4'), ('b8', '<f4')]
        np_pc = np.frombuffer(msg.data, dtype=np_dtype)
        points[:,0]=np_pc['x']
        points[:,1]=np_pc['y']
        points[:,2]=np_pc['z']
        points[:,3]=np_pc['i']
        points[:,3]/=255.0

    return points