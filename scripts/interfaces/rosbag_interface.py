from interfaces.rosnumpy import rosmsg_to_numpy
import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message

def get_rosbag_options(path, storage_id, serialization_format='cdr'):
    storage_options = rosbag2_py.StorageOptions(
        uri=path, storage_id=storage_id)

    converter_options = rosbag2_py.ConverterOptions(
        input_serialization_format=serialization_format,
        output_serialization_format=serialization_format)

    return storage_options, converter_options

class Bag2PointCloud:
    def __init__(self, path, topic):
        storage_options, converter_options = get_rosbag_options(path, 'sqlite3')
        self.reader = rosbag2_py.SequentialReader()
        self.reader.open(storage_options, converter_options)
        topic_types = self.reader.get_all_topics_and_types()

        # Create a map for quicker lookup
        self.type_map = {topic_types[i].name: topic_types[i].type for i in range(len(topic_types))}

        # Set filter for topic of string type
        storage_filter = rosbag2_py.StorageFilter(topics=[topic])
        self.reader.set_filter(storage_filter)
    
    def __del__(self):
        self.reader.__exit__()

    def next_pc(self, get='msg', lidar='velo'):
        while self.reader.has_next():
            (topic, data, ts) = self.reader.read_next()
            msg_type = get_message(self.type_map[topic])
            msg = deserialize_message(data, msg_type)
            points = rosmsg_to_numpy(msg, lidar)

            if get=='msg':
                yield msg, points
            else:
                yield ts, points

# user = 'prabin'
# handle = Bag2PointCloud('/home/'+user+'/Datasets/KITTI/kitti_2011_09_26_drive_0005_synced', '/kitti/velo/pointcloud')
# for _, points in handle.next_pc():
#     print(points)
#     break