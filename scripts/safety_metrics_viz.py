import json
import time
import matplotlib.pyplot as plt
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['font.size'] = 12

import numpy as np
from matplotlib.animation import FuncAnimation
from metrics.dsa_metrics import calculate_ttc
from metrics.intersection_point import intersection_point

from interfaces.rosbag_interface import Bag2PointCloud
from sensor_msgs.msg import PointCloud2
from visualization_msgs.msg import MarkerArray
from marker_gen import get_markers_from_tracks, get_text_markers_from_tracks
from pc_tools.pc_tools import transform_points, downsample_points
from interfaces.rosnumpy import numpy_to_rosmsg
import threading
import rclpy
rclpy.init()
node = rclpy.create_node('safety_metrics')
thread = threading.Thread(target=rclpy.spin, args=(node, ), daemon=True)
thread.start()
user = 'user'
pc_pub = node.create_publisher(PointCloud2, 'live_points', 10)
marker_pub = node.create_publisher(MarkerArray, 'markers', 10)

def get_overlap_tracklets(leader_tracklet, follower_tracklet):
    assert len(leader_tracklet) > 0 and len(follower_tracklet) > 0
    leader_stamps = [track['stamp'] for track in leader_tracklet]
    follower_stamps = [track['stamp'] for track in follower_tracklet]
    if leader_stamps[0] > follower_stamps[-1] or follower_stamps[0] > leader_stamps[-1]:
        raise Exception('No Overlap Between Tracklets')
    l_idx, f_idx = 0, 0
    while abs(leader_stamps[l_idx] - follower_stamps[f_idx]) > 1e-6:
        if leader_stamps[l_idx] > follower_stamps[f_idx]:
            f_idx += 1
        else:
            l_idx += 1
    leader_start_idx, follower_start_idx = l_idx, f_idx
    while l_idx < len(leader_stamps) and f_idx < len(follower_stamps):
        l_idx += 1
        f_idx += 1
    leader_end_idx, follower_end_idx = l_idx, f_idx
    leader_tracklet = [leader_tracklet[idx] for idx in range(leader_start_idx, leader_end_idx)]
    follower_tracklet = [follower_tracklet[idx] for idx in range(follower_start_idx, follower_end_idx)]
    return leader_tracklet, follower_tracklet

def evaluate_car_following_ttc(leader_tracklet, follower_tracklet, plot=False, warmup_offset=5):
    assert abs(leader_tracklet[0]['stamp'] - follower_tracklet[0]['stamp']) < 1e-6 and \
           abs(leader_tracklet[-1]['stamp'] - follower_tracklet[-1]['stamp']) < 1e-6 and \
           len(leader_tracklet) == len(follower_tracklet)
    ttc = [] 
    for idx in range(warmup_offset, len(leader_tracklet)):
        leader_pos = np.asarray(leader_tracklet[idx]['position'])
        follower_pos = np.asarray(follower_tracklet[idx]['position'])
        leader_vel = np.asarray(leader_tracklet[idx]['velocity'])
        follower_vel = np.asarray(follower_tracklet[idx]['velocity'])
        delta_pos = np.linalg.norm(leader_pos-follower_pos)
        delta_vel = np.linalg.norm(leader_vel-follower_vel)
        ttc.append(calculate_ttc(delta_pos, delta_vel))
    stamps = np.asarray([track['stamp'] for track in leader_tracklet]) - leader_tracklet[0]['stamp']
    stamps = stamps[warmup_offset:]
    
    if plot:
        plt.figure()
        plt.title('TTC (Time to Collision)')
        plt.plot(stamps, ttc)
        plt.ylabel('Time (s)')
        plt.xlabel('Step (0.05s)')
        plt.show()

    return ttc

def evaluate_intersection_metrics(leader_tracklet, follower_tracklet, info):
    start_ts = leader_tracklet[0]['stamp']
    global points
    if 'bag_handle' in info:
        bag_handle = info['bag_handle']
        for msg, points in bag_handle.next_pc(lidar='hesai_bag'):
            ts = msg.header.stamp.sec + msg.header.stamp.nanosec / 10 ** 9
            if abs(ts-start_ts) < 1e-4:
                break

    stamps = np.asarray([track['stamp'] for track in leader_tracklet]) - start_ts
    leader_pos = np.vstack([np.asarray(track['position']) for track in leader_tracklet])
    leader_vel = np.vstack([np.asarray(track['velocity']) for track in leader_tracklet])
    follower_pos = np.vstack([np.asarray(track['position']) for track in follower_tracklet])
    follower_vel = np.vstack([np.asarray(track['velocity']) for track in follower_tracklet])
    leader_dims = np.vstack([np.asarray(track['scale']) for track in leader_tracklet])
    follower_dims = np.vstack([np.asarray(track['scale']) for track in follower_tracklet])
    leader_orientation = np.asarray([track['orientation'] for track in leader_tracklet])
    follower_orientation = np.asarray([track['orientation'] for track in follower_tracklet])

    fig, ax = plt.subplots()
    ax.set_title(f'Leader-ID: {info["leader_id"]} | Follower-ID: {info["follower_id"]}', fontsize=20)
    traj1 = ax.plot([], [], 'b-', label='Leader')[0]
    traj2 = ax.plot([], [], 'r-', label='Follower')[0]
    arrow1 = ax.arrow(x=leader_pos[0,0], 
        y=leader_pos[0,1], 
        dx=leader_vel[0,0], 
        dy=leader_vel[0,1], width = 0.25)
    arrow2 = ax.arrow(x=follower_pos[0,0], 
        y=follower_pos[0,1], 
        dx=follower_vel[0,0], 
        dy=follower_vel[0,1], width = 0.25)
    proj1 = ax.plot(leader_pos[0,0], leader_pos[0,1], linestyle='dashed')[0]
    proj2 = ax.plot(follower_pos[0,0], follower_pos[0,1], linestyle='dashed')[0]
    obj1 = ax.plot([], [], 'g-', linewidth=1.5)[0]
    obj2 = ax.plot([], [], 'g-', linewidth=1.5)[0]
    cross = ax.plot([], [], 'mo', linewidth=3)[0]
    pet_text = ax.text(None, None, None)
    mdse_text = ax.text(None, None, None)
    mdse_text.set_x(5)
    mdse_text.set_y(25)
    ax.legend(loc='upper right')
    ax.grid()
    ax.axis('equal')
    ax.set(xlim=[5, 45], ylim=[-20, 20], xlabel='X (meters)', ylabel='Y (meters)')

    pet_track = []
    se_track = []
    sev_track = []

    def update(idx, horizon=5):
        # if idx==0:
        #     time.sleep(10)
        if 'bag_handle' in info:
            global points
            points = downsample_points(points, voxel_size=0.10) # Mill Ave
            points = transform_points(points, xyz=(0., 0., 7.5), rpy=(0.45, -0.005, 1.4)) # Mill Ave
            pc_pub.publish(numpy_to_rosmsg(points, 'PandarSwift'))
            colors = []
            leader_track = [*leader_pos[idx].tolist(), 
                            leader_orientation[idx], 
                            *leader_dims[idx].tolist(), 
                            info["leader_id"]]
            colors.append((1.,0.,0.,0.5) if int(info["leader_id"])%10==1 else (0.,1.,0.,0.5))
            follower_track = [*follower_pos[idx].tolist(), 
                              follower_orientation[idx], 
                              *follower_dims[idx].tolist(), 
                              info["follower_id"]]
            colors.append((1.,0.,0.,0.5) if int(info["follower_id"])%10==1 else (0.,1.,0.,0.5))
            marker_pub.publish(get_markers_from_tracks([leader_track, follower_track], 'PandarSwift', 0.5, colors))
            marker_pub.publish(get_text_markers_from_tracks([leader_track, follower_track], 'PandarSwift', 0.5))
            _, points = next(bag_handle.next_pc(lidar='hesai_bag'))

        traj1.set_xdata(leader_pos[:idx,0])
        traj1.set_ydata(leader_pos[:idx,1])
        traj2.set_xdata(follower_pos[:idx,0])
        traj2.set_ydata(follower_pos[:idx,1])

        arrow1.set_data(x=leader_pos[idx,0], 
            y=leader_pos[idx,1], 
            dx=leader_vel[idx,0], 
            dy=leader_vel[idx,1])
        arrow2.set_data(x=follower_pos[idx,0], 
            y=follower_pos[idx,1], 
            dx=follower_vel[idx,0], 
            dy=follower_vel[idx,1], width = 0.5)

        def bbox_line(pos, vel, dims):
            bbox_hl, bbox_hw = dims[idx,0]/2, dims[idx,1]/2
            bbox = np.asarray([(-bbox_hw, bbox_hl),
                    (bbox_hw, bbox_hl),
                    (bbox_hw, -bbox_hl),
                    (-bbox_hw, -bbox_hl),
                    (-bbox_hw, bbox_hl),
                    (0,0),
                    (bbox_hw, bbox_hl)]).T
            dy, dx = vel[idx,1], vel[idx,0]
            theta = np.arctan2(dy,dx) - np.pi/2
            RotM = np.asarray([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
            bbox = RotM @ bbox
            bbox[0,:] += pos[idx,0]
            bbox[1,:] += pos[idx,1]
            return bbox.T

        bbox1 = bbox_line(leader_pos, leader_vel, leader_dims)
        obj1.set_xdata(bbox1[:,0])
        obj1.set_ydata(bbox1[:,1])
        bbox2 = bbox_line(follower_pos, follower_vel, follower_dims)
        obj2.set_xdata(bbox2[:,0])
        obj2.set_ydata(bbox2[:,1])

        leader_back_point = np.mean(bbox1[2:4,:], axis=0)
        follower_back_point = np.mean(bbox2[2:4,:], axis=0)
        proj1.set_xdata((leader_back_point[0], leader_pos[idx,0]+leader_vel[idx,0]*horizon))
        proj1.set_ydata((leader_back_point[1], leader_pos[idx,1]+leader_vel[idx,1]*horizon))
        proj2.set_xdata((follower_back_point[0], follower_pos[idx,0]+follower_vel[idx,0]*horizon))
        proj2.set_ydata((follower_back_point[1], follower_pos[idx,1]+follower_vel[idx,1]*horizon))

        cross_point = intersection_point((leader_back_point[0], leader_back_point[1]),
                                         (leader_pos[idx,0]+leader_vel[idx,0]*horizon,
                                         leader_pos[idx,1]+leader_vel[idx,1]*horizon),
                                         (follower_back_point[0], follower_back_point[1]),
                                         (follower_pos[idx,0]+follower_vel[idx,0]*horizon,
                                         follower_pos[idx,1]+follower_vel[idx,1]*horizon))

        if cross_point:
            cross.set_xdata(cross_point[0])
            cross.set_ydata(cross_point[1])
            pet_text.set_x(cross_point[0]+1.0)
            pet_text.set_y(cross_point[1])

            ### PET Calculation ########################
            t1 = np.linalg.norm(np.asarray(cross_point)-leader_pos[idx,:2]) /\
                np.linalg.norm(leader_vel[idx,:2])
            t2 = np.linalg.norm(np.asarray(cross_point)-follower_pos[idx,:2]) /\
                np.linalg.norm(follower_vel[idx,:2])
            # modified PET according to the paper "An adaptive peer-to-peer collision warning system"
            pet = round(abs(t2-t1),3)
            pet_track.append([stamps[idx], pet])
            pet_text.set_text(s=str(pet)+' s')

            ### MDSE Calculation #######################
            # RSS parameters for MSDV metric (Conservative category) for human drivers
            # react_time = 1.9
            # long_max_acc = 5.9
            # long_min_decel = 4.1
            # RSS parameters for MSDV metric (NDS category) for automated vehicles
            react_time = 0.2
            long_max_acc = 1.8
            long_min_decel = 3.6
            # RSS parameters for MSDV metric (Aggressive category) for automated vehicles
            # react_time = 0.5
            # long_max_acc = 4.1
            # long_min_decel = 4.6
            vel_long = np.linalg.norm(leader_vel[idx])
            d_long_intersect_min = vel_long*react_time + \
                0.5*long_max_acc*react_time**2 + \
                    (vel_long+react_time*long_max_acc)**2 /\
                        (2*long_min_decel)
            d_long_intersect = \
                np.linalg.norm(np.asarray(cross_point)-leader_pos[idx,:2])
            se_track.append([stamps[idx], d_long_intersect, d_long_intersect_min])
            print(f'Min safe long distance: {d_long_intersect_min} | Current distance to conflict point: {d_long_intersect}')
            if d_long_intersect_min > d_long_intersect:
                # AV should react here because the conflict point lies within 5sec future window
                # and conflict point lies within the safety envelope of leader
                # although in our dataset we have only human drivers and no AVs
                # we assume the follower to be an AV and use NDS parameters to evaluate metrics
                # if ADS clones human behavior then the analysis is meaningful
                print('MDSE Violation')
                mdse_text.set_text('MDSE Violation')
                sev_track.append([stamps[idx], 1.0])
            else:
                mdse_text.set_text(None)
                sev_track.append([stamps[idx], 0.0])
        else:
            cross.set_xdata([])
            cross.set_ydata([])
            pet_text.set_text(None)
            mdse_text.set_text(None)

        return traj1, traj2, arrow1, arrow2, proj1, proj2, obj1, obj2, cross, pet_text

    anim = FuncAnimation(fig=fig, func=update, frames=stamps.shape[0], interval=1, repeat=False)
    plt.show()

    pet_track = np.asarray(pet_track)
    se_track = np.asarray(se_track)
    sev_track = np.asarray(sev_track)
    start_idx = 1

    plt.figure()
    plt.plot(pet_track[start_idx:,0]-pet_track[start_idx,0], pet_track[start_idx:,1],  '-o', label='PET')
    plt.plot(pet_track[start_idx:,0]-pet_track[start_idx,0], 1.5*np.ones((len(pet_track)-start_idx,)),  '--r', label='Threshold')
    plt.title('Post Encroachment Time (PET)')
    plt.xlabel('Time step (0.05s)')
    plt.ylabel('Time (s)')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
          fancybox=False, shadow=False, ncol=2)
    plt.tight_layout()
    plt.show()

    plt.figure()
    plt.plot(se_track[start_idx:,0]-se_track[start_idx,0], se_track[start_idx:,1], '-ro', label='Distance to CP')
    plt.plot(se_track[start_idx:,0]-se_track[start_idx,0], se_track[start_idx:,2], '-bo', label='Distance to SE')
    plt.title('Minimum Distance Safety Envelope (MDSE)')
    plt.xlabel('Time step (0.05s)')
    plt.ylabel('Distance (m)')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
          fancybox=False, shadow=False, ncol=2)
    plt.tight_layout()
    plt.show()

    plt.figure()
    plt.plot(sev_track[start_idx:,0]-sev_track[start_idx,0], sev_track[start_idx:,1], '-o')
    plt.title('Minimum Distance Safety Envelope Infringement (MDSEI)')
    plt.xlabel('Time step (0.05s)')
    plt.ylabel('MDSEI')
    plt.show()
    

data_id = 4
with open(f'hesai_millave_{data_id}_pvrcnn.json', 'r') as fp:
    # Subject vehicle is the AV, Prioritized vehicle is the AV, Follower is the AV
    # Other is vehicle/pedestrian, Non prioritized is vehicle/pedestrian, Leader is the vehicle/pedestrian

    bag_handle = Bag2PointCloud(f'/home/{user}/Datasets/Hesai/hesai_mill_ave_{data_id}', '/hesai/pandar_points')

    ## hesai_millave_4_pvrcnn
    
    # potential collision
    leader = '4621' 
    follower = '4740'

    # leader = '10590' 
    # follower = '8110'

    # leader = '1500' 
    # follower = '1360'
    
    ## hesai_millave_2_pvrcnn

    # leader = '980' 
    # follower = '1090'

    # leader = '431' 
    # follower = '1170'

    ## hesai_millave_1_pvrcnn

    # leader = '1510' 
    # follower = '0'

    # potential collision
    # leader = '1300' 
    # follower = '2220'
    
    tracklets = json.load(fp)
    leader_tracklet = tracklets[leader]
    follower_tracklet = tracklets[follower]
    leader_tracklet, follower_tracklet = get_overlap_tracklets(leader_tracklet, follower_tracklet)

    # Metrics Evaluation
    # evaluate_car_following_ttc(leader_tracklet, follower_tracklet, plot=True)
    evaluate_intersection_metrics(leader_tracklet, follower_tracklet, 
    {'leader_id':leader, 'follower_id':follower, 'bag_handle': bag_handle})
