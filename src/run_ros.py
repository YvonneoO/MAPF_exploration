#!/usr/bin/python3
import os
import rospy
import argparse
import glob
from pathlib import Path
from cbs import CBSSolver
from pbs import PBSSolver
from single_agent_planner import get_sum_of_cost
# from mapf_exploration.msg import robotStates
from map_manager.msg import robotStates
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path as waypoints
from mapf_exploration.msg import PathArray 
from mapf_exploration.msg import gridInfoGain
from matplotlib.patches import Circle, Rectangle
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
import math
import copy
import time
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

        # simple office

        
        # industry office
        self.grid_size = 10.0
        self.map_size_x = 40
        self.map_size_y = 40
        self.map_origin_x = -self.map_size_x / 2.0
        self.map_origin_y = -self.map_size_y / 2.0

        script_dir = os.path.dirname(os.path.abspath(__file__))
        file = os.path.join(script_dir, instance)
        
        if not os.path.exists(file):
            raise FileExistsError(f"Instance file '{file}' does not exist.")
        else:
            self.file = file
    
        print("***Import an instance***")
        self.my_map= self.import_mapf_instance(file)
        self.starts = [(0,0) for i in range(self.num_agents)]
        self.robots_ready = [False for i in range(self.num_agents)]
        #self.print_mapf_instance(self.my_map, self.starts, self.goals)
        
        self.information_map = copy.deepcopy(self.my_map)
        for i in range(len(self.information_map)):
            for j in range(len(self.information_map[0])):
                self.information_map[i][j] = 1 - self.information_map[i][j]
        
        rospy.loginfo("MAPF initialization complete. Ready to start planning")
        
        self.pose_subscriber = rospy.Subscriber(robotState_topic, robotStates, self.robotStateCB)
        
        # subscribe information gain of the large grid regions calculated by the explorer
        self.info_gain_subscriber = rospy.Subscriber('/grid_info_gain', gridInfoGain, self.infoGainCB)
        
        # TO DO:
        # ? subscribe dependent mode replan flag?
        # replan due to ugv unable to complete exploring the region, force-set this cbs_waypoint (the region) to uav
        # TO DO:
        
        # cbs waypoints represents the center of large grid regions, 
        self.waypoints_visualizer = rospy.Publisher('/cbs_waypoints', MarkerArray, queue_size=10)
        self.path_publisher = rospy.Publisher(expRegion_topic, PathArray, queue_size=100)
        
        
        self.timer = rospy.Timer(rospy.Duration(5.0), self.solve)
        self.vis_timer = rospy.Timer(rospy.Duration(0.5), self.vis_cb)
        self.replan_timer = rospy.Timer(rospy.Duration(1.0), self.replan_cb)
        print("current time:", rospy.Time.now())

    def replan_cb(self, event=None):
        # if no path is found, replan
        if self.paths is None:
            self.replan = True
            return
        
        for (i,j) in self.starts:
            if self.information_map[i][j] <0.3:
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
            ColorRGBA(1.0, 1.0, 0.0, transparent),  # Yellow
            ColorRGBA(1.0, 0.0, 1.0, transparent),  # Magenta
            ColorRGBA(0.0, 1.0, 1.0, transparent),  # Cyan
            ColorRGBA(1.0, 0.0, 0.0, transparent),  # Red
            ColorRGBA(0.0, 1.0, 0.0, transparent),  # Green
            ColorRGBA(0.0, 0.0, 1.0, transparent),  # Blue
        ]

        marker_array = MarkerArray()

        for i, point_list in enumerate(self.paths):
            color = colors[i % len(colors)]

            # Marker for CUBE_LIST (waypoints as cubes)
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
                p.x = self.map_origin_x + point[0] * self.grid_size + self.grid_size / 2
                p.y = self.map_origin_y + point[1] * self.grid_size + self.grid_size / 2
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
        # self.starts[self.robotState.robot_id-1] = (
        #                                             int((self.map_size_x/2-self.robotState.position[0])//self.grid_size), 
        #                                             int((self.map_size_y/2-self.robotState.position[1])//self.grid_size)
        #                                            )
        grid_x = int((self.robotState.position[0] - self.map_origin_x) // self.grid_size)
        grid_y = int((self.robotState.position[1] - self.map_origin_y) // self.grid_size)

        self.starts[self.robotState.robot_id - 1] = (grid_x, grid_y)

        self.robots_ready[self.robotState.robot_id-1] = True
        # print("received start position: ", self.starts)
        for (i, ready) in enumerate(self.robots_ready):
            if not ready:
                print("robot %i is not ready" % (i+1))
                return
        
        
    def infoGainCB(self, info_gain_msg: gridInfoGain):
        if self.information_map[info_gain_msg.grid_idx_x][info_gain_msg.grid_idx_y] >= info_gain_msg.percent_info_gain:
            self.information_map[info_gain_msg.grid_idx_x][info_gain_msg.grid_idx_y] = info_gain_msg.percent_info_gain
        #     print("real time information map: ", self.information_map)
        # print("receive info gain : %i, %i, %f" % (info_gain_msg.grid_idx_x, info_gain_msg.grid_idx_y, info_gain_msg.percent_info_gain))
        # self.replan = True

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
                rospy.loginfo("Publishing path array")
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