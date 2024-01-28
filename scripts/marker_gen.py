from visualization_msgs.msg import MarkerArray, Marker
from tf_transformations import quaternion_from_euler
from geometry_msgs.msg import Quaternion
from rclpy.clock import Clock
from rclpy.duration import Duration

def get_markers_from_mmdet(boxes_3d, frame_id, p_dt):
    marker_array = MarkerArray()
    for i in range(boxes_3d.shape[0]):
        marker = Marker()
        marker.header.frame_id = frame_id
        marker.header.stamp = Clock().now().to_msg()
        marker.ns = "bounding_box"
        marker.id = i
        marker.type = Marker.CUBE
        marker.action = Marker.ADD
        marker.pose.position.x = boxes_3d[i][0]
        marker.pose.position.y = boxes_3d[i][1]
        marker.pose.position.z = boxes_3d[i][2] + boxes_3d[i][5]/2
        quat = quaternion_from_euler(0,0,boxes_3d[i][6])
        marker.pose.orientation = Quaternion(x=quat[0], y=quat[1], z=quat[2], w=quat[3])
        marker.scale.x = boxes_3d[i][3]
        marker.scale.y = boxes_3d[i][4]
        marker.scale.z = boxes_3d[i][5]
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.color.a = 0.5
        marker.lifetime = Duration(seconds=p_dt).to_msg()
        marker_array.markers.append(marker)
    return marker_array

def get_markers_from_tracks(tracks, frame_id, p_dt, colors=None):
    marker_array = MarkerArray()
    if colors == None:
        colors = [(0.,1.,0.,0.5),]*tracks.shape[0]
    for track, color in zip(tracks,colors):
        marker = Marker()
        marker.header.frame_id = frame_id
        marker.header.stamp = Clock().now().to_msg()
        marker.ns = "bounding_box"
        marker.id = int(track[7])
        marker.type = Marker.CUBE
        marker.action = Marker.ADD
        marker.pose.position.x = track[0]
        marker.pose.position.y = track[1]
        marker.pose.position.z = track[2]
        quat = quaternion_from_euler(0,0,track[3])
        marker.pose.orientation = Quaternion(x=quat[0], y=quat[1], z=quat[2], w=quat[3])
        marker.scale.x = track[4]
        marker.scale.y = track[5]
        marker.scale.z = track[6]
        marker.color.r = color[0]
        marker.color.g = color[1]
        marker.color.b = color[2]
        marker.color.a = color[3]
        marker.lifetime = Duration(seconds=p_dt).to_msg()
        marker_array.markers.append(marker)
    return marker_array

def get_text_markers_from_tracks(tracks, frame_id, p_dt):
    marker_array = MarkerArray()
    for track in tracks:
        marker = Marker()
        marker.header.frame_id = frame_id
        marker.header.stamp = Clock().now().to_msg()
        marker.ns = "identify_box"
        marker.id = int(track[7])
        marker.type = Marker.TEXT_VIEW_FACING
        marker.action = Marker.ADD
        marker.pose.position.x = track[0]
        marker.pose.position.y = track[1]
        marker.pose.position.z = track[2] + 1.0
        quat = quaternion_from_euler(0,0,0)
        marker.pose.orientation = Quaternion(x=quat[0], y=quat[1], z=quat[2], w=quat[3])
        marker.text = str(int(track[7]))
        marker.scale.x = 2.
        marker.scale.y = 2.
        marker.scale.z = 2.
        marker.color.r = 0.0
        marker.color.g = 0.0
        marker.color.b = 1.0
        marker.color.a = 1.0
        marker.lifetime = Duration(seconds=p_dt).to_msg()
        marker_array.markers.append(marker)
    return marker_array
