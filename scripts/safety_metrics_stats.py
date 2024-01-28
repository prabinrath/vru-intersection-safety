import json
from collections import defaultdict
import numpy as np
from metrics.intersection_point import intersection_point
import cv2

def bbox_line(pos, vel, dims):
    bbox_hl, bbox_hw = dims[0]/2, dims[1]/2
    bbox = np.asarray([(-bbox_hw, bbox_hl),
            (bbox_hw, bbox_hl),
            (bbox_hw, -bbox_hl),
            (-bbox_hw, -bbox_hl),
            (-bbox_hw, bbox_hl),
            (0,0),
            (bbox_hw, bbox_hl)]).T
    dy, dx = vel[1], vel[0]
    theta = np.arctan2(dy,dx) - np.pi/2
    RotM = np.asarray([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    bbox = RotM @ bbox
    bbox[0,:] += pos[0]
    bbox[1,:] += pos[1]
    return bbox.T

def project_to_img(robot_pos, local_range, img_half_dim):
    x = int((2*(robot_pos[0]-local_range[0][0])/(local_range[0][1]-local_range[0][0]))*img_half_dim)
    y = int((2*(robot_pos[1]-local_range[1][0])/(local_range[1][1]-local_range[1][0]))*img_half_dim)
    return img_half_dim*2-y, img_half_dim*2-x

def pet_metric(cross_point, leader_pos, leader_vel, follower_pos, follower_vel):
    ### PET Calculation ########################
    t1 = np.linalg.norm(cross_point-leader_pos) /\
        np.linalg.norm(leader_vel)
    t2 = np.linalg.norm(cross_point-follower_pos) /\
        np.linalg.norm(follower_vel)
    # modified PET according to the paper "An adaptive peer-to-peer collision warning system"
    pet = round(abs(t2-t1),3)
    return pet

def mdse_metric(cross_point, leader_pos, leader_vel):
    ### MDSE Calculation #######################
    # RSS parameters for MSDV metric (Conservative category) for human drivers
    # react_time = 1.9
    # long_max_acc = 5.9
    # long_min_decel = 4.1
    # RSS parameters for MSDV metric (NDS category) for automated vehicles
    # react_time = 0.2
    # long_max_acc = 1.8
    # long_min_decel = 3.6
    # RSS parameters for MSDV metric (Aggressive category) for automated vehicles
    react_time = 0.5
    long_max_acc = 4.1
    long_min_decel = 4.6
    vel_long = np.linalg.norm(leader_vel)
    d_long_intersect_min = vel_long*react_time + \
        0.5*long_max_acc*react_time**2 + \
            (vel_long+react_time*long_max_acc)**2 /\
                (2*long_min_decel)
    d_long_intersect = \
        np.linalg.norm(cross_point-leader_pos)
    # print(f'Min safe long distance: {d_long_intersect_min} | Current distance to conflict point: {d_long_intersect}')
    if d_long_intersect_min > d_long_intersect:
        # AV should react here because the conflict point lies within 5sec future window
        # and conflict point lies within the safety envelope of leader
        # although in our dataset we have only human drivers and no AVs
        # we assume the follower to be an AV and use NDS parameters to evaluate metrics
        # if ADS clones human behavior then the analysis is justified
        # print('MDSE Violation')
        return True
    return False

def calculate_metrics(cross_point, leader_pos, leader_vel, follower_pos, follower_vel):
    petv = pet_metric(cross_point, leader_pos, leader_vel, follower_pos, follower_vel) < 1.5
    mdsev = mdse_metric(cross_point, leader_pos, leader_vel)
    return petv, mdsev


# Subject vehicle is the AV, Prioritized vehicle is the AV, Follower is the AV
# Other is vehicle/pedestrian, Non prioritized is vehicle/pedestrian, Leader is the vehicle/pedestrian

user = 'user'
ids = [1,4]

total_veh_veh_situs = 0
total_veh_veh_petv = 0
total_veh_veh_msdev = 0
total_veh_veh_common = 0
total_ped_veh_situs = 0
total_ped_veh_petv = 0
total_ped_veh_msdev = 0
total_ped_veh_common = 0

for data_id in ids:
    horizon = 5
    min_tracklet_len = 3
    warmup_frames = 5
    local_range = [[-5,65],[-35,35]]
    img_half_dim = 200
    car_car_unq_situs = set()
    ped_car_unq_situs = set()
    car_car_unq_petv = set()
    ped_car_unq_petv = set()
    car_car_unq_mdsev = set()
    ped_car_unq_mdsev = set()
    num_ped_tracklets = 0
    num_car_tracklets = 0

    anim = False

    with open(f'hesai_millave_{data_id}_pvrcnn.json', 'r') as fp:
        raw_tracklets = json.load(fp)
        tracklets = {}
        for key in raw_tracklets:
            tracklet = raw_tracklets[key]
            warmup = min(warmup_frames, len(tracklet)-1)
            if abs(tracklet[warmup]['stamp'] - tracklet[-1]['stamp']) > min_tracklet_len:
                if tracklet[0]['label'] == 'ped':
                        num_ped_tracklets += 1
                elif tracklet[0]['label'] == 'car':
                    num_car_tracklets += 1
                tracklets[key] = tracklet
        
        rt_data_stream = defaultdict(list)
        for key in tracklets:
            tracklet = tracklets[key]
            for track in tracklet:
                track['key'] = key
                rt_data_stream[track['stamp']].append(track)
        
        for objs in rt_data_stream.values():
            if anim:
                canvas = np.ones((img_half_dim*2,img_half_dim*2,3), dtype=np.uint8)*255
                for obj in objs:
                    center = project_to_img(obj['position'], local_range, img_half_dim)
                    if obj['label'] == 'car':
                        cv2.circle(canvas, center, 5, (0, 255, 0), -1) 
                    elif obj['label'] == 'ped':
                        cv2.circle(canvas, center, 5, (0, 0, 255), -1) 
                    bbox = bbox_line(obj['position'], obj['velocity'], obj['scale'])
                    back_point = np.mean(bbox[2:4,:], axis=0)
                    back_point = project_to_img(back_point, local_range, img_half_dim)
                    proj_point = project_to_img((obj['position'][0]+obj['velocity'][0]*horizon,
                                                obj['position'][1]+obj['velocity'][1]*horizon), local_range, img_half_dim)
                    cv2.arrowedLine(canvas, back_point, proj_point, (255, 0, 0), 2)

            # this loop is O(N^2) as it is for offline computation. online version has the N(log(N)) algorithm
            # that will be released in future. 
            for i in range(len(objs)):
                for j in range(i+1,len(objs)):
                    if not (objs[i]['label'] == 'ped' and objs[j]['label'] == 'ped'):
                        bbox1 = bbox_line(objs[i]['position'], objs[i]['velocity'], objs[i]['scale'])
                        bbox2 = bbox_line(objs[j]['position'], objs[j]['velocity'], objs[j]['scale'])
                        leader_back_point = np.mean(bbox1[2:4,:], axis=0)
                        follower_back_point = np.mean(bbox2[2:4,:], axis=0)
                        cross_point = intersection_point((leader_back_point[0], leader_back_point[1]),
                                                (objs[i]['position'][0]+objs[i]['velocity'][0]*horizon,
                                                objs[i]['position'][1]+objs[i]['velocity'][1]*horizon),
                                                (follower_back_point[0], follower_back_point[1]),
                                                (objs[j]['position'][0]+objs[j]['velocity'][0]*horizon,
                                                objs[j]['position'][1]+objs[j]['velocity'][1]*horizon))
                        if cross_point:
                            if objs[i]['label'] == 'ped' or objs[j]['label'] == 'ped':
                                if objs[i]['label'] == 'ped':
                                    leader = objs[i]
                                    follower = objs[j]
                                else:
                                    leader = objs[j]
                                    follower = objs[i]
                                petv, mdesv = calculate_metrics(np.array(cross_point), 
                                                                np.array(leader['position'][:2]),
                                                                np.array(leader['velocity'][:2]),
                                                                np.array(follower['position'][:2]),
                                                                np.array(follower['velocity'][:2]))
                                
                                ped_car_unq_situs.add(f'{leader["key"]}-{follower["key"]}')
                                if petv:
                                    ped_car_unq_petv.add(f'{leader["key"]}-{follower["key"]}')
                                if mdesv:
                                    ped_car_unq_mdsev.add(f'{leader["key"]}-{follower["key"]}')
                            else:
                                leader = objs[i]
                                follower = objs[j]
                                petv, mdesv = calculate_metrics(np.array(cross_point), 
                                                                np.array(leader['position'][:2]),
                                                                np.array(leader['velocity'][:2]),
                                                                np.array(follower['position'][:2]),
                                                                np.array(follower['velocity'][:2]))
                                car_car_unq_situs.add(f'{leader["key"]}-{follower["key"]}')
                                if petv:
                                    car_car_unq_petv.add(f'{leader["key"]}-{follower["key"]}')
                                if mdesv:
                                    car_car_unq_mdsev.add(f'{leader["key"]}-{follower["key"]}')
                            if anim:
                                cross_point = project_to_img(cross_point, local_range, img_half_dim)
                                cv2.circle(canvas, cross_point, 5, (255, 0, 255), -1) 
            if anim:
                cv2.imshow('canvas', canvas)
                cv2.waitKey(50)

    ped_car_common_v = ped_car_unq_petv.intersection(ped_car_unq_mdsev)
    car_car_common_v = car_car_unq_petv.intersection(car_car_unq_mdsev)

    total_veh_veh_situs += len(car_car_unq_situs)
    total_veh_veh_petv += len(car_car_unq_petv)
    total_veh_veh_msdev += len(car_car_unq_mdsev)
    total_veh_veh_common += len(car_car_common_v)

    total_ped_veh_situs += len(ped_car_unq_situs)
    total_ped_veh_petv += len(ped_car_unq_petv)
    total_ped_veh_msdev += len(ped_car_unq_mdsev)
    total_ped_veh_common += len(ped_car_common_v)

    print(f'------ Stats for data id {data_id} ------')
    print(f'number of car tracklets: {num_car_tracklets}, number of ped tracklets: {num_ped_tracklets}')
    print(f'car car situations: {len(car_car_unq_situs)}, ped car situations: {len(ped_car_unq_situs)}')
    print(f'ped car pet violations: {len(ped_car_unq_petv)}, ped car mdse violations: {len(ped_car_unq_mdsev)}')
    print(f'ped car common violations: {len(ped_car_common_v)}')
    print(f'car car pet violations: {len(car_car_unq_petv)}, car car mdse violations: {len(car_car_unq_mdsev)}')
    print(f'car car common violations: {len(car_car_common_v)}')
    print('------------------------------------------')

print(f'{total_veh_veh_situs}, {total_veh_veh_petv}, {total_veh_veh_msdev}, {total_veh_veh_common}')
print(f'{total_ped_veh_situs}, {total_ped_veh_petv}, {total_ped_veh_msdev}, {total_ped_veh_common}')