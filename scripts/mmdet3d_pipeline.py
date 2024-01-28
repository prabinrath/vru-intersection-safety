from sensor_msgs.msg import PointCloud2
from visualization_msgs.msg import MarkerArray
import numpy as np
import rclpy
import threading

from mmdet3d.apis import init_model, inference_detector

import time
import json
from collections import defaultdict
from interfaces.rosbag_interface import Bag2PointCloud
from mot_3d.mot import MOTModel
from mot_3d.frame_data import FrameData
from mot_3d.data_protos import BBox
from marker_gen import get_markers_from_tracks, get_text_markers_from_tracks
from pc_tools.pc_tools import transform_points, downsample_points
from interfaces.rosnumpy import numpy_to_rosmsg

rclpy.init()
node = rclpy.create_node('evsts_pipeline')
thread = threading.Thread(target=rclpy.spin, args=(node, ), daemon=True)
thread.start()
user = 'user'
handle = Bag2PointCloud(f'/home/{user}/Datasets/Hesai/hesai_mill_ave_1', '/hesai/pandar_points')

frame_id = 'PandarSwift'
lidar = 'hesai_bag' # type of data packet for rosnumpy conversions

pc_pub = node.create_publisher(PointCloud2, 'live_points', 10)
marker_pub = node.create_publisher(MarkerArray, 'markers', 10)
rate = node.create_rate(10)

# config_file = 'mmdet3d/configs/pointpillars/pointpillars_hv_secfpn_8xb6-160e_kitti-3d-3class.py'
# checkpoint_file = 'mmdet3d/checkpoints/hv_pointpillars_secfpn_6x8_160e_kitti-3d-3class_20220301_150306-37dc2420.pth'

config_file = 'scripts/mmdet3d/configs/pv_rcnn/pv_rcnn_8xb2-80e_kitti-3d-3class.py'
checkpoint_file = 'scripts/mmdet3d/pv_rcnn_8xb2-80e_kitti-3d-3class_20221117_234428-b384d22f.pth'

model = init_model(config_file, checkpoint_file, device='cuda:0')

p_dt = 0.5
configs = {
            "running": {
                "covariance": "default",
                "score_threshold": 0.4,
                "tracker": "cp_plus",
                "max_age_since_update": {
                    "cp_plus": 4
                    },
                "min_hits_to_birth": {
                    "immortal": 3,
                    "cp_plus": 3
                    },
                # "match_type": "greedy",
                # "asso": "m_dis",
                "match_type": "bipartite",
                "asso": "giou",
                "asso_thres": {
                    "iou": 0.9,
                    "giou": 1.5,
                    "m_dis": 5
                    }
            }
          }
mgr_car = MOTModel(configs)

configs = {
            "running": {
                "covariance": "default",
                "score_threshold": 0.0,
                "tracker": "cp_plus",
                "max_age_since_update": {
                    "cp_plus": 5
                    },
                "min_hits_to_birth": {
                    "immortal": 3,
                    "cp_plus": 1
                    },
                # "match_type": "greedy",
                # "asso": "m_dis",
                "match_type": "bipartite",
                "asso": "giou",
                "asso_thres": {
                    "iou": 0.9,
                    "giou": 1.5,
                    "m_dis": 5
                    }
            }
          }
mgr_ped = MOTModel(configs)

tracks_dict = defaultdict(list)

def get_tracks(mgr, boxes_3d, scores_3d, labels_3d, stamp, label):
    frame = []
    for i in range(boxes_3d.shape[0]):
        x, y, z, sx, sy, sz, yaw = boxes_3d[i][0], boxes_3d[i][1], boxes_3d[i][2] + boxes_3d[i][5]/2, boxes_3d[i][3], boxes_3d[i][4], boxes_3d[i][5], boxes_3d[i][6]
        frame.append(np.array([x, y, z, yaw, sx, sy, sz, scores_3d[i]]))
    data = FrameData(dets=frame, ego=None, time_stamp=stamp, aux_info={'is_key_frame':True}, det_types=labels_3d)
    track_output = mgr.frame_mot(data)
    tracks = []
    for idx, (bbox, id, state, _) in enumerate(track_output):
        if 'alive' in state:
            track_state = BBox.bbox2array(bbox)
            if label == 'car': 
                key = id*10 + 0
            elif label == 'ped':
                key = id*10 + 1
            else:
                key = id*10 + 2
            tracks.append(np.append(track_state[:7],key))
            tracks_dict[key].append({'stamp': stamp,
                                    'position': track_state[:3].tolist(),
                                    'orientation': track_state[3],
                                    'velocity': mgr.trackers[idx].motion_model.kf.x[7:,0].tolist()})
    return tracks

count = 0
for msg, points in handle.next_pc(lidar=lidar): 
    count+=1
    stamp = msg.header.stamp.sec + msg.header.stamp.nanosec / 10 ** 9
    start_time = time.perf_counter()

    points = downsample_points(points, voxel_size=0.10) # Mill Ave
    points = transform_points(points, xyz=(0., 0., 7.5), rpy=(0.45, -0.005, 1.4)) # Mill Ave

    result, data = inference_detector(model, points)
    result = result.pred_instances_3d
    boxes_3d, scores_3d, labels_3d = result['bboxes_3d'].tensor.cpu().numpy(), result['scores_3d'].cpu().numpy(), result['labels_3d'].cpu().numpy()
    # print(boxes_3d, scores_3d, labels_3d)

    car_tracks = get_tracks(mgr_car, boxes_3d[labels_3d==2], scores_3d[labels_3d==2], labels_3d[labels_3d==2], stamp, 'car')
    # car_tracks = get_tracks(mgr_car, boxes_3d[labels_3d==0], scores_3d[labels_3d==0], labels_3d[labels_3d==0], stamp, 'car')
    if len(car_tracks)>0:
        tracks = np.vstack(car_tracks)
        print(tracks.shape[0])
        colors = [(0.,1.,0.,0.5),]*tracks.shape[0]
        marker_pub.publish(get_markers_from_tracks(tracks, frame_id, p_dt, colors))
        marker_pub.publish(get_text_markers_from_tracks(tracks, frame_id, p_dt))
    
    ped_tracks = get_tracks(mgr_ped, boxes_3d[labels_3d==0], scores_3d[labels_3d==0], labels_3d[labels_3d==0], stamp, 'ped')
    if len(ped_tracks)>0:
        tracks = np.vstack(ped_tracks)
        print(tracks.shape[0])
        colors = [(1.,0.,0.,0.5),]*tracks.shape[0]
        marker_pub.publish(get_markers_from_tracks(tracks, frame_id, p_dt, colors))
        marker_pub.publish(get_text_markers_from_tracks(tracks, frame_id, p_dt))

    # boxes_3d = boxes_3d[scores_3d>0].astype(np.float64)
    # if boxes_3d.shape[0] > 0:
    #     print(boxes_3d.shape[0])
    #     marker_pub.publish(get_markers_from_mmdet(boxes_3d, frame_id, p_dt))

    pc_pub.publish(numpy_to_rosmsg(points, frame_id))
    duration = time.perf_counter()-start_time
    node.get_logger().info('Published Sample '+ str(count)+' in '+str(round(duration,3))+' secs')
    rate.sleep()

with open('tracking_results.json', 'w') as fp:
    json.dump(tracks_dict, fp)