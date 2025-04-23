#!/usr/bin/python3
import os
import rospy
import copy
import time
from pathlib import Path
from cbs import CBSSolver
from single_agent_planner import get_sum_of_cost
from map_manager.msg import robotStates
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path as waypoints
from mapf_exploration.msg import PathArray, gridInfoGain
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA

class MAPF_Publisher:
    def __init__(self, solver: str, instance: str, batch: bool, robotState_topic: str, expRegion_topic: str):
        """
        :param 
        """

        rospy.loginfo(f"Instance Path received: {instance}")
        rospy.loginfo(f"Solver argument received: {solver}")
        self.instance = instance
        self.solver = solver
        self.batch = batch
        self.num_agents = 0
        self.paths = None
        self.replan = True

        self.grid_size = 10.0
        self.map_size_x = 42
        self.map_size_y = 42
        self.map_origin_x = -21.0
        self.map_origin_y = -21.0
        
        # exploration status threshold
        self.grid_explored_thres = 0.3
        self.grid_unexplored_thres = 0.7
        self.localIG_thres = 0.1
        self.stuck_duration_thres = 50.0
        
        self.current_region = {} # robot_id -> (x, y)
        self.region_enter_time = {} # robot_id -> rospy.Time
        self.region_has_left = {}  # robot_id -> bool
        

        script_dir = os.path.dirname(os.path.abspath(__file__))
        file = os.path.join(script_dir, instance)
        
        if not os.path.exists(file):
            raise FileExistsError(f"Instance file '{file}' does not exist.")
        else:
            self.file = file
    
        print("***Import an instance***")
        self.my_map= self.import_mapf_instance(file)
        self.map_height = len(self.my_map)
        self.map_width = len(self.my_map[0]) if self.map_height > 0 else 0

        self.starts = [(0,0) for i in range(self.num_agents)]
        self.robots_ready = [False for i in range(self.num_agents)]
        self.restricted_assignments = {i: set() for i in range(self.num_agents)}
        #self.print_mapf_instance(self.my_map, self.starts, self.goals)
        
        self.information_map = copy.deepcopy(self.my_map)
        self.information_map = [
            [{'global': 1.0 if not self.my_map[i][j] else 0.0, 'local': 1.0}
            for j in range(len(self.my_map[0]))]
            for i in range(len(self.my_map))
        ]
        
        rospy.loginfo("MAPF initialization complete. Ready to start planning")
        
        self.pose_subscriber = rospy.Subscriber(robotState_topic, robotStates, self.robotStateCB)
        
        # subscribe information gain of the large grid regions calculated by the explorer
        self.info_gain_subscriber = rospy.Subscriber('/grid_info_gain', gridInfoGain, self.infoGainCB)

        self.waypoints_visualizer = rospy.Publisher('/cbs_waypoints', MarkerArray, queue_size=10)
        self.path_publisher = rospy.Publisher(expRegion_topic, PathArray, queue_size=100)

        # Store the most recent valid path
        self.last_valid_path_array = None
        
        # Add a separate publisher timer with higher frequency
        self.publish_timer = rospy.Timer(rospy.Duration(0.5), self.publish_paths)
        
        self.timer = rospy.Timer(rospy.Duration(5.0), self.solve)
        self.vis_timer = rospy.Timer(rospy.Duration(0.5), self.vis_cb)
        self.replan_timer = rospy.Timer(rospy.Duration(1.0), self.replan_cb)
        
        print("current time:", rospy.Time.now())

    def publish_paths(self, event=None):
        """Continuously publish the most recent valid path"""
        if self.last_valid_path_array is not None:
            # Add timestamp
            self.last_valid_path_array.header.stamp = rospy.Time.now()
            self.path_publisher.publish(self.last_valid_path_array)
            
    def replan_cb(self, event=None):
        # if no path is found, replan
        if self.paths is None:
            rospy.loginfo("[CBS Replan Trigger] No path found, replan.")
            self.replan = True
            return
        
        for robot_id, path in enumerate(self.paths):
            # if any robot finishes its path, replan
            all_explored = True
            for (x, y) in path:
                cell = self.information_map[x][y]
                if cell['global'] >= self.grid_explored_thres:  # not fully explored yet
                    all_explored = False
                    break
            if all_explored:
                rospy.loginfo(f"[CBS Replan Trigger] Robot {robot_id+1} has finished all assigned regions.")
                self.replan = True
                return
            
            # if any robot stucks in current exp region, not able to complete, replan
            # for (x, y) in path:
            #     cell = self.information_map[x][y]
            #     if cell['global'] >= 0.8 and cell['local'] <= 0.005:
            #         rospy.logwarn(f"[CBS Reassignment] Robot {robot_id+1} cannot finish Region ({x},{y}), reassigning.")
            #         self.restricted_assignments[robot_id].add((x, y))
            #         self.replan = True
            #         return
            
        # if any robot stucks in current exp region, not able to complete, replan
        for robot_id in range(self.num_agents):
            region = self.current_region.get(robot_id)
            enter_time = self.region_enter_time.get(robot_id)
            has_left = self.region_has_left.get(robot_id, False)

            if region is None or enter_time is None or has_left:
                continue

            duration = (rospy.Time.now() - enter_time).to_sec()
            
            if self.robots_ready[robot_id] and duration < self.stuck_duration_thres:
                continue

            x, y = region
            cell = self.information_map[x][y]
            if cell['global'] >= self.grid_unexplored_thres and cell['local'] <= self.localIG_thres:
                rospy.logwarn(f"[CBS Reassignment] Robot {robot_id+1} stuck in Region ({x},{y}) for {duration:.1f}s. Reassigning.")
                self.restricted_assignments[robot_id].add((x, y))
                self.replan = True
                return

        
        # for any of the path, the information gain of any of the waypoint is lower than 0.1, replan
        # self.replan = True
        # for path in self.paths:
        #     for waypoint in path:
        #         if self.information_map[waypoint[0]][waypoint[1]] > 0.2:
        #             self.replan = False
        #             return

    def vis_cb(self, event=None):
        if self.paths is None:
            return

        transparent = 0.5
        colors = [
            ColorRGBA(1.0, 1.0, 0.0, transparent),  # Yellow #ugv
            ColorRGBA(0.0, 0.0, 1.0, transparent),  # Blue #uav
            ColorRGBA(0.0, 1.0, 0.0, transparent),  # Green
            ColorRGBA(1.0, 0.0, 0.0, transparent),  # Red
            ColorRGBA(1.0, 1.0, 0.0, transparent),  # Yellow
            ColorRGBA(1.0, 0.0, 1.0, transparent),  # Magenta
            ColorRGBA(0.0, 1.0, 1.0, transparent),  # Cyan
        ]

        marker_array = MarkerArray()

        for i, point_list in enumerate(self.paths):
            color = colors[i % len(colors)]

            # Marker for CUBE_LIST (waypoints / regions as cubes)
            cube_marker = Marker()
            cube_marker.header.frame_id = "map"
            cube_marker.id = i
            cube_marker.header.stamp = rospy.Time.now()
            cube_marker.type = Marker.CUBE_LIST
            cube_marker.action = Marker.ADD
            cube_marker.scale.x = self.grid_size
            cube_marker.scale.y = self.grid_size
            cube_marker.scale.z = 2.0
            cube_marker.color = color

            # Marker for LINE_STRIP (connecting path)
            line_marker = Marker()
            line_marker.header.frame_id = "map"
            line_marker.id = i + 1000  # Different namespace
            line_marker.header.stamp = rospy.Time.now()
            line_marker.type = Marker.LINE_STRIP
            line_marker.action = Marker.ADD
            line_marker.scale.x = 0.3  # Line width
            line_marker.color = color

            for point in point_list:
                p = Point()
                p.x = (point[0] + 0.5) * self.grid_size + self.map_origin_x
                p.y = (point[1] + 0.5) * self.grid_size + self.map_origin_y
                p.z = 1.0
                cube_marker.points.append(p)
                line_marker.points.append(p)

            marker_array.markers.append(cube_marker)
            marker_array.markers.append(line_marker)

        self.waypoints_visualizer.publish(marker_array)

    def robotStateCB(self, robotState_msg: robotStates):
        self.robotState = robotState_msg
        # print("robot state")
        # print(self.robotState.robot_id)
        # print(self.robotState.ready)
        # print(self.robotState.position[0])
        # print(self.robotState.position[1])

        robot_id = self.robotState.robot_id - 1
        grid_x = int((self.robotState.position[0] - self.map_origin_x) // self.grid_size)
        grid_y = int((self.robotState.position[1] - self.map_origin_y) // self.grid_size)
        current_grid = (grid_x, grid_y)
        
        # Initialize tracking
        if robot_id not in self.current_region:
            self.current_region[robot_id] = current_grid
            self.region_enter_time[robot_id] = rospy.Time.now()
            self.region_has_left[robot_id] = False
        else:
            last_region = self.current_region[robot_id]

            if current_grid != last_region:
                self.region_has_left[robot_id] = True  # robot left the region
                self.current_region[robot_id] = current_grid
                self.region_enter_time[robot_id] = rospy.Time.now()

        self.starts[robot_id] = current_grid
        self.robots_ready[robot_id] = True
        
        # print("received start position: ", self.starts)
        for (i, ready) in enumerate(self.robots_ready):
            if not ready:
                print("robot %i is not ready" % (i+1))
                return

    def infoGainCB(self, info_gain_msg: gridInfoGain):
        x_idx = info_gain_msg.grid_idx_x
        y_idx = info_gain_msg.grid_idx_y

        if 0 <= x_idx < len(self.information_map) and 0 <= y_idx < len(self.information_map[0]):
            cell = self.information_map[x_idx][y_idx]
            if cell['global'] == 0.0:
                ROS_WARN("skip updates for obstacles defined in the initial 2d grid map")
            cell['global'] = min(cell['global'], info_gain_msg.global_percent_info_gain)
            cell['local'] = min(cell['local'], info_gain_msg.local_percent_info_gain)
        else:
            rospy.logwarn(f"Grid indices out of range: ({x_idx}, {y_idx}). Map size: {len(self.information_map)}x{len(self.information_map[0])}")

    def solve(self, event=None): 
        """ callback function for publisher """
        # rospy.loginfo("In solver callback")
        # check if all robot sready
        for (i, ready) in enumerate(self.robots_ready):
            if not ready:
                print("robot %i is not ready for replan" % (i+1))
                return
        if self.replan:
            start_time = time.time()
            rospy.loginfo("Solver triggered")
            result_file = open("results.csv", "w", buffering=1)

            if self.solver == "CBS":
                print("***Run CBS***")
                information_map_copy = copy.deepcopy(self.information_map)
                print("information map copy: ", information_map_copy)
                cbs = CBSSolver(self.my_map, self.starts, information_map_copy)
                cbs.restricted_assignments = self.restricted_assignments
                paths = cbs.explore_environment()
                rospy.loginfo("CBS solver complete")
                path_array = PathArray()
                for path in paths:
                    print("path:")
                    path_waypoints = waypoints()
                    for waypoint in path:
                        point = PoseStamped()
                        print(waypoint)
                        point.pose.position.x = waypoint[0]
                        point.pose.position.y = waypoint[1]
                        path_waypoints.poses.append(point)
                    path_array.PathArray.append(path_waypoints)
                # path_array.header.stamp = img_timestamp   
                # rospy.loginfo("Publishing path array")
                # self.path_publisher.publish(path_array)
                        
                # Store the path for continuous publishing
                self.last_valid_path_array = path_array
                # Initial publish
                self.path_publisher.publish(path_array)
            else:
                raise RuntimeError("Unknown solver!")
            self.paths = paths
            cost = get_sum_of_cost(paths)
            result_file.write("{},{}\n".format(self.file, cost))

            if not self.batch:
                print("***Test paths on a simulation***")
                # animation = Animation(self.my_map, self.starts, self.goals, paths)
                # animation.show()

            result_file.close()
            self.replan = False
            end_time = time.time()  # End time after the function execution
            print("Time taken to execute the function: ", end_time - start_time, "seconds")

    def print_mapf_instance(self, my_map, starts, goals):
        rospy.loginfo('Start locations')
        self.print_locations(my_map, starts)
        #rospy.loginfo('Goal locations')
        #self.print_locations(my_map, goals)


    def print_locations(self, my_map, locations):
        starts_map = [[-1 for _ in range(len(my_map[0]))] for _ in range(len(my_map))]
        for i in range(len(locations)):
            starts_map[locations[i][0]][locations[i][1]] = i
        to_print = '\n'
        to_print += ''
        for x in range(len(my_map)):
            for y in range(len(my_map[0])):
                if starts_map[x][y] >= 0:
                    to_print += str(starts_map[x][y]) + ' '
                elif my_map[x][y]:
                    to_print += '@ '
                else:
                    to_print += '. '
            to_print += '\n'
        rospy.loginfo(to_print)


    def import_mapf_instance(self, filename):
        f = Path(filename)
        if not f.is_file():
            raise BaseException(filename + " does not exist.")
        f = open(filename, 'r')
        # first line: #rows #columns
        line = f.readline()
        rows, columns = [int(x) for x in line.split(' ')]
        rows = int(rows)
        columns = int(columns)
        # #rows lines with the map
        my_map = []
        for r in range(rows):
            line = f.readline()
            my_map.append([])
            for cell in line:
                if cell == '@':
                    my_map[-1].append(True)
                elif cell == '.':
                    my_map[-1].append(False)
        # #agents
        line = f.readline()
        self.num_agents = int(line)
        # #agents lines with the start/goal positions
        # starts = []
        # goals = []
        # for a in range(num_agents):
        #     line = f.readline()
        #     sx, sy, gx, gy = [int(x) for x in line.split(' ')]
        #     starts.append((sx, sy))
        #     goals.append((gx, gy))
        f.close()
        return my_map

if __name__ == '__main__':
    SOLVER = "CBS"

    rospy.init_node('mapf_exploration_node', anonymous=True)
    solver = rospy.get_param('~solver', SOLVER)
    # instance = rospy.get_param('~instance', 'instances/simple_office.txt')
    instance = rospy.get_param('~instance', 'instances/industry_office.txt')
    batch = rospy.get_param('~batch', False)
    robotState_topic = '/robot_states'
    expRegion_topic = '/exp_region_cbs'


    publisher = MAPF_Publisher(solver, instance, batch, robotState_topic, expRegion_topic)

    rospy.spin()