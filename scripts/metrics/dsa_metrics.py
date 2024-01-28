#!/usr/bin/env python

#####################################################
## DSA Metrics Calculation for Scenarios
##
## Code by Shujauddin Rahimi
## Based on safety_metrics_calculation.py
## Written by Maria S. Elli
#####################################################

import numpy as np
from math import sqrt, pow, sin, cos

def calculate_ci(d_lon, d_lat_r, d_lat_l, collision_threshold = 0.2):
    '''
    Collision Incident     
    '''
    ci_lon = d_lon < collision_threshold
    ci_lat_r = d_lat_r < collision_threshold
    ci_lat_l = d_lat_l < collision_threshold

    ci = ci_lon & (ci_lat_r | ci_lat_l)

    ci = int(ci)

    return ci

def calculate_mdse_lon_violation(d_lon, v_1, v_2, a_max, b_min, rho, b_max):
    # Vehicles traveling in the same direction
    # d_lon is the recorded longitudinal difference between vehicles 1 and 2

    first_term = v_1 * rho
    second_term = 0.5 * a_max * rho **2
    third_term = ((v_1 + rho * a_max)**2)/(2 * b_min)
    fourth_term = (v_2**2) / (2 * b_max)

    d_lon_min = max(first_term + second_term + third_term - fourth_term, 0)

    if (d_lon <= d_lon_min):
        return [1, d_lon_min]
    else:
        return [0, d_lon_min]
    
def calculate_mdse_lat_violation(d_lat_r, d_lat_l, mu, v_1_lat, rho_1, a_1_lat_min_decel, a_1_lat_max_accel, v_2_lat, rho_2, a_2_lat_max_accel, a_2_lat_min_decel):
    # Vehicle 1 is to the left of Vehicle 2

    first_term = ((2 * v_1_lat + rho_1 * a_1_lat_max_accel)/2) * rho_1
    second_term = ((v_1_lat + rho_1 * a_1_lat_max_accel) ** 2) / (2 * a_1_lat_min_decel)
    third_term = ((2 * v_2_lat - rho_2 * a_2_lat_max_accel)/2) * rho_2
    fourth_term = ((v_2_lat - rho_2 * a_2_lat_max_accel) ** 2) / (2 * a_2_lat_min_decel)

    d_lat_min = mu + max((first_term + second_term - (third_term - fourth_term)), 0)

    if (d_lat_r <= d_lat_min or d_lat_l <= d_lat_min):
        return [1, d_lat_min]
    else:
        return [0, d_lat_min]

# Used for SEI
def calculate_mdsei(mdse_lon_v, mdse_lat_v):
    if (mdse_lon_v and mdse_lat_v):
        return 1
    else:
        return 0

# Used for SEV
def calculate_mdsev(msdv_lon, msdv_lat, b_lon_max, a_lat_max, other_lon_acc, other_lat_acc):
    '''
    Minimum Safe Distance Violation 
    
    Happens when both, the longitudinal and lateral msd are unsafe
    '''
    ### MSDV' if both directions are unsafe
    dangerous_situation = msdv_lon & msdv_lat

    ### if other vehicle did not brake harder than b_max, then, the ego vehicle may have caused the dangerous situation 
    other_brake_less_than_b_max = other_lon_acc >= b_lon_max
    
    ### if other vehicle did not steer faster a_lat_max, then, the ego vehicle may have caused the dangerous situation 
    other_acc_less_than_a_lat_max = abs(other_lat_acc) <= abs(a_lat_max)
    
    other_within_assumptions = other_acc_less_than_a_lat_max & other_brake_less_than_b_max
    
    msdv = int(dangerous_situation & other_within_assumptions)

    return msdv

# Used for SER
def calculate_mdser(d_lon_min, d_lon):
    # Currently, only focused on car following scenarios

    return d_lon / d_lon_min

def calculate_mdse():
    # in meters
    d_lon = 0
    d_lat = 0

    # in meters per second
    v_lon_object = 0
    v_lat_object = 0

    v_lon_subject = 0
    v_lat_subject = 0

    # in seconds, subject vehicle response time
    subject_vrt = 0
    object_vrt = 0

    # in meters per second squared
    a_lon_max_accel_subject = 0

    a_lat_min_decel_subject = 0
    a_lat_max_accel_subject = 0

    a_lon_min_decel_subject = 0
    a_lon_max_decel_subject = 0
    
    a_lon_max_decel_object = 0
    a_lon_max_accel_object = 0

    a_lon_min_decel_object = 0

    a_lat_max_accel_object = 0
    a_lat_min_decel_object = 0

    mu = 0

    #Minimum longitudinal distance between two vehicles traveling in the same lon direction
    d_lon_min_same  =   ( (v_lon_subject * subject_vrt) 
                        + (0.5 * a_lon_max_accel_subject * subject_vrt ^ 2)
                        + ((v_lon_subject + subject_vrt*a_lon_max_accel_subject)^2 
                        / (2 * a_lon_min_decel_subject))
                        - ((v_lon_object)^2
                        / (2 * a_lon_max_decel_object))
                        )
    
    d_lon_min_opp   =   ( (2 * v_lon_subject + subject_vrt * a_lon_max_accel_subject) / 2 * subject_vrt
                        + ((v_lon_subject + subject_vrt * a_lon_max_accel_subject)^2 / (2 * a_lon_min_decel_subject))
                        + ((2 * abs(v_lon_object) + object_vrt * a_lon_max_accel_object) / (2 * object_vrt))
                        + ((abs(v_lon_object) + object_vrt * a_lon_max_accel_object)^2 / (2 * a_lon_min_decel_object))
                        )
    
    d_lat_min       = ( mu
                        + (2 * (v_lat_subject + subject_vrt * a_lat_max_accel_subject)/2 * subject_vrt)
                        + ((v_lat_subject + subject_vrt * a_lat_max_accel_subject)^2 / 2 * a_lat_min_decel_subject)
                        - ((2 * v_lat_object - object_vrt * a_lat_max_accel_object) / 2 * object_vrt
                        - (v_lat_object - object_vrt * a_lat_max_accel_object) ^ 2 / (2 * a_lat_min_decel_object)) 
                        )
    
    d_lon_min_intersect =   ( v_lon_subject * subject_vrt
                            + (1/2 * a_lon_max_accel_subject * subject_vrt ^ 2)
                            + ((v_lon_subject + subject_vrt * a_lon_max_accel_subject) ^ 2) / (2 * a_lon_min_decel_subject)
                            )

def calculate_ttc(delta_pos, delta_vel, ttc_max = 10):
    '''
    Time To Collision [s]
    '''
    ttc = ttc_max

    if delta_vel <= 0:
        ttc = ttc_max
    else:
        ttc = delta_pos/delta_vel
        ttc = min(ttc_max, ttc)
        # if ttc < 0:
        #     ttc = ttc_max

    return ttc

def calculate_mttc(delta_pos, delta_vel, delta_acc, mttc_max = 10):
    '''
    Modified Time To Collision [s]
    '''
    mttc = mttc_max

    v = delta_vel
    a = delta_acc
    d = delta_pos
    
    if a == 0 and  v > 0:
        mttc = min(d / v, mttc_max)

    if a == 0 and v <= 0:
        mttc = mttc_max

    if (v**2 + 2*a*d) < 0 :
        mttc = mttc_max
    elif a != 0:
        try:
            t_1 = (-v -(v**2 + 2*a*d)**0.5)/a
            t_2 = (-v +(v**2 + 2*a*d)**0.5)/a
            
            if t_1 > 0 and t_2 > 0:
                if t_1 >= t_2:
                    mttc = t_2
                elif t_1 < t_2:
                    mttc = t_1
            elif t_1 > 0 and t_2 <= 0:
                mttc = t_1
            elif t_1 <= 0 and t_2 > 0:
                mttc = t_2

        ### exception for 'ValueError: negative number cannot be raised to a fractional power'
        ### exception for 'TypeError: '>' not supported between instances of 'complex' and 'int'
        except (ValueError, TypeError) as e:
            # print(repr(e))
            mttc = mttc_max
        
    mttc = min(mttc, mttc_max)

    return mttc

def calculate_pet_curve(ego_center_x, ego_center_y, ego_yaw, ego_length, other_center_x, other_center_y, other_yaw, other_length, timestamp, pet_max = np.nan):
    '''
    Post Encroachment Time Curve [s] (postprocessing metric)
    '''
    df_length = len(timestamp)
    
    ego_front_bumper_x = ego_center_x + ego_yaw.apply(cos)*ego_length/2
    ego_front_bumper_y = ego_center_y + ego_yaw.apply(sin)*ego_length/2

    other_rear_bumper_x = other_center_x - other_yaw.apply(cos)*other_length/2
    other_rear_bumper_y = other_center_y - other_yaw.apply(sin)*other_length/2

    diff_arr = np.zeros(df_length)
    min_PET = np.array([])

    for n in range(df_length):
        diff_arr = np.zeros(df_length)
        local_front_bumper_x = ego_front_bumper_x.loc[n]
        local_front_bumper_y = ego_front_bumper_y.loc[n]
        timestamp_n = timestamp.loc[n]
        
        for i in range(df_length):
            diff_arr[i] = sqrt(pow(local_front_bumper_x - other_rear_bumper_x.loc[i], 2) + pow(local_front_bumper_y - other_rear_bumper_y.loc[i], 2))

        conflict_point_list = list(timestamp[(diff_arr > -1) & (diff_arr < 1)].index)
        
        if len(conflict_point_list) != 0:
            PET_list = np.zeros(len(conflict_point_list))
            
            list_index_count = 0
            
            for c in conflict_point_list:
                PET_list[list_index_count] = timestamp_n - timestamp.loc[c]
                PET_list = abs(PET_list)
                list_index_count += 1
                
            min_PET = np.append(min_PET, min(PET_list))
        
        else:
            min_PET = np.append(min_PET, np.nan)  # Replace np.nan with any other default value for continuous visualization
        
    return min_PET

def calculate_thw(distance, speed, thw_max = 10):
    '''
    Time Headway [s]
    '''
    if speed > 0 :
        thw = (distance / speed)
    else:
        thw = thw_max
    thw = min (thw, thw_max)

    return thw

def calculate_lsv(waypoints, current_yaw):
    waypoint_delta_x = waypoints[1][0] - waypoints[0][0]
    waypoint_delta_y = waypoints[1][1] - waypoints[0][1]
    waypoint_heading = np.arctan(waypoint_delta_y/waypoint_delta_x)
    heading_error_mod = divmod((waypoint_heading - current_yaw), np.pi)[1]

    if heading_error_mod > np.pi/2 and heading_error_mod < np.pi:
        heading_error_mod -= np.pi
        
    return heading_error_mod