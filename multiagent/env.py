import json
import os
import numpy as np
import random
from collections import defaultdict
import cv2
import torch
import shapely
import shapely.geometry
from shapely.geometry import Point, Polygon, LineString, MultiPoint
from shapely.ops import nearest_points

from multiagent.cityreferobject import get_city_refer_objects
from multiagent.dataset.generate import generate_episodes_from_mturk_trajectories
from multiagent.dataset.mturk_trajectory import load_mturk_trajectories
from multiagent.mapdata import MAP_BOUNDS
from multiagent.maps.landmark_nav_map import LandmarkNavMap
from multiagent.observation import cropclient
from multiagent.space import Pose4D, modulo_radians, Point2D
from typing import List, Dict, Callable, Tuple


def _convert_contours_to_centroids(landmarks: List[List[Point2D]]):
    centroids = []
    for lm in landmarks:
        contour = np.array(lm)
        # closed_contour = np.array(contour + [contour[0]], dtype=np.float32).reshape(-1, 1, 2)
        # M = cv2.moments(closed_contour)
        # if M["m00"] != 0:  # 避免除以零
        #     cx = M["m10"] / M["m00"]
        #     cy = M["m01"] / M["m00"]
        #     centroids.append((cx, cy))
        # print(contour.shape)
        centroid = np.mean(contour, axis=0)
        centroids.append(centroid)
    return centroids


def compute_iou(a, b):
    a = np.array(a)  # quadrilateral two-dimensional coordinate representation
    poly1 = Polygon(
        a).convex_hull  # python quadrilateral object, will automatically calculate four points, the last four points in the order of: top left bottom right bottom right top left top
    # print(Polygon(a).convex_hull)  # you can print to see if this is the case

    b = np.array(b)
    poly2 = Polygon(b).convex_hull
    # print(Polygon(b).convex_hull)

    union_poly = np.concatenate((a, b))  # Merge two box coordinates to become 8*2
    # print(union_poly)
    # print(MultiPoint(union_poly).convex_hull)  # contains the smallest polygon point of the two quadrilaterals
    if not poly1.intersects(poly2):  # If the two quadrilaterals do not intersect
        iou = 0
    else:
        try:
            inter_area = poly1.intersection(poly2).area  # intersection area
            # print(inter_area)
            # union_area = poly1.area + poly2.area - inter_area
            union_area = MultiPoint(union_poly).convex_hull.area
            # print(union_area)
            if union_area == 0:
                iou = 0
            # iou = float(inter_area)/(union_area-inter_area)  #wrong
            iou = float(inter_area) / union_area
            # iou=float(inter_area) /(poly1.area+poly2.area-inter_area)
            # The source code gives two ways to calculate IOU, the first one is: intersection part / area of the smallest polygon containing two quadrilaterals
            # The second one: intersection/merge (common way to calculate IOU of rectangular box)
        except shapely.geos.TopologicalError:
            print('shapely.geos.TopologicalError occured, iou set to 0')
            iou = 0
    return iou


def get_direction(start, end):
    vec = np.array(end) - np.array(start)
    _angle = 0
    #          90
    #      135    45
    #     180  .    0
    #      225   -45 
    #          270
    if vec[1] > 0:  # lng is postive
        _angle = np.arctan(vec[0] / vec[1]) / 1.57 * 90
    elif vec[1] < 0:
        _angle = np.arctan(vec[0] / vec[1]) / 1.57 * 90 + 180
    else:
        if np.sign(vec[0]) == 1:
            _angle = 90
        else:
            _angle = 270
    _angle = (360 - _angle + 90) % 360
    return _angle


def name_the_direction(_angle):
    if _angle > 337.5 or _angle < 22.5:
        return 'north'
    elif np.abs(_angle - 45) <= 22.5:
        return 'northeast'
    elif np.abs(_angle - 135) <= 22.5:
        return 'southeast'
    elif np.abs(_angle - 90) <= 22.5:
        return 'east'
    elif np.abs(_angle - 180) <= 22.5:
        return 'south'
    elif np.abs(_angle - 315) <= 22.5:
        return 'northwest'
    elif np.abs(_angle - 225) <= 22.5:
        return 'southwest'
    elif np.abs(_angle - 270) <= 22.5:
        return 'west'


class CityNavBatch(torch.utils.data.IterableDataset):
    def __init__(self, split, args,
                 batch_size=64, seed=0, rank=0, world_size=1):
        self.split = split
        self.data = []
        self.args = args

        self.nav_maps = []
        # for anno_file in anno_files:
        #     with jsonlines.open(anno_file, 'r') as f:
        #         for item in f:
        #             self.data.append(item)
        cropclient.load_image_cache()
        objects = get_city_refer_objects()
        full_data = generate_episodes_from_mturk_trajectories(
            objects, load_mturk_trajectories(split, 'all', args.altitude))

        random.seed(seed)
        if self.split == 'train_seen':
            random.shuffle(full_data)

        # 分片：每个进程处理不同部分
        self.data = full_data[rank::world_size]
        # self.data = full_data

        self.ix = 0
        self.batch_size = batch_size
        self.rank = rank
        self.world_size = world_size

    def size(self):
        return len(self.data)

    # TODO: find where it is used and then write it
    # def _get_gt_trajs(self, data):
    #     return {x['instr_id']: (x['scan'], x['end_panos']) for x in data if 'end_panos' in x}

    def next_batch(self):

        batch_size = self.batch_size

        for ix in range(0, len(self.data), batch_size):
            batch = self.data[ix: ix + batch_size]
            if len(batch) < batch_size:
                # random.shuffle(self.data)
                ix = batch_size - len(batch)
                batch += self.data[:ix]

            self.batch = batch
            used_map_names = []
            for i in range(batch_size):
                used_map_names.append(self.batch[i].map_name)

            # Get the max instruction length
            max_instruction_length = 0
            for i in range(batch_size):
                if len(self.batch[i].target_description) > max_instruction_length:
                    max_instruction_length = len(self.batch[i].target_description)
            self.max_instruction_length = max_instruction_length

            yield used_map_names

    def __iter__(self):
        return self.next_batch()

    def normalize_position(self, pos: Point2D, map_name: str, map_meters: float):
        return (pos.x - MAP_BOUNDS[map_name].x_min) / map_meters, (MAP_BOUNDS[map_name].y_max - pos.y) / map_meters

    def unnormalize_position(self, normalized_xy: Tuple[float, float], map_name: str, map_meters: float):
        nx, ny = normalized_xy
        return Point2D(nx * map_meters + MAP_BOUNDS[map_name].x_min, MAP_BOUNDS[map_name].y_max - ny * map_meters)

    def normalize_to_real(self, id: int, map_name: str, map_meters: float):
        return Point2D(MAP_BOUNDS[map_name].x_min + (float(id) // self.args.grid_size + 0.5) * map_meters,
                       MAP_BOUNDS[map_name].y_max - (float(id) % self.args.grid_size + 0.5) * map_meters)

    # TODO: provide whole environment per time step
    def _get_obs(self, poses: List[Pose4D] = None, random_direction = False):
        obs = []

        if poses is None:
            poses = [episode.start_pose for episode in self.batch]
            # poses = [episode.trajectory[max(0, len(episode.trajectory) - 20)] for episode in self.batch]
            self.nav_maps = [
                LandmarkNavMap(
                    episode.map_name, self.args.map_shape, self.args.map_pixels_per_meter,
                    episode.description_landmarks)
                for episode in self.batch]

        for i in range(self.batch_size):
            episode = self.batch[i]

            # if poses is None:
            #     # poses = episode.start_pose
            #
            #     self.nav_maps.append(
            #         LandmarkNavMap(
            #             episode.map_name, self.args.map_shape, self.args.map_pixels_per_meter,
            #             episode.description_landmarks, episode.description_target, episode.description_surroundings))
            #
            #     current_position = episode.trajectory[max(0, len(episode.trajectory) - 20)].xy
            # else:
            if random_direction:
                tem = random.random()
                if tem < 0.5:
                    poses[i] = Pose4D(poses[i].xyz.x, poses[i].xyz.y, poses[i].xyz.z,
                                  random.random() * (2 * 3.141592653589793) - 3.141592653589793)
                # poses[i].yaw = random.random() * (2 * 3.141592653589793) - 3.141592653589793
            current_position = poses[i].xyz
            normalized_goal_xys = self.normalize_position(episode.target_position, episode.map_name,
                                                          self.args.map_meters)

            # if normalized_goal_xys[0] > 1:
            #     normalized_goal_xys[0] =
            normalized_goal_grid = np.floor(np.array(normalized_goal_xys) * self.args.grid_size)

            normalized_cur_pos = self.normalize_position(current_position, episode.map_name, self.args.map_meters)
            normalized_cur_grid = np.floor(np.array(normalized_cur_pos) * self.args.grid_size)

            if normalized_goal_grid[1] >= self.args.grid_size:
                normalized_goal_grid[1] = self.args.grid_size - 1
            if normalized_goal_grid[0] >= self.args.grid_size:
                normalized_goal_grid[0] = self.args.grid_size - 1
            normalized_goal_id = normalized_goal_grid[0] * self.args.grid_size + normalized_goal_grid[1]
            if (normalized_goal_id >= self.args.grid_size ** 2) or (normalized_goal_id < 0):
                print(normalized_goal_xys)


            normalized_cur_pos = self.normalize_position(current_position, episode.map_name, self.args.map_meters)
            normalized_cur_grid = np.floor(np.array(normalized_cur_pos) * self.args.grid_size)
            if normalized_cur_grid[1] >= self.args.grid_size:
                normalized_cur_grid[1] = self.args.grid_size - 1
            if normalized_cur_grid[0] >= self.args.grid_size:
                normalized_cur_grid[0] = self.args.grid_size - 1
            if normalized_cur_grid[0] < 0:
                normalized_cur_grid[0] = 0
            if normalized_cur_grid[1] < 0:
                normalized_cur_grid[1] = 0
            normalized_pos_id = normalized_cur_grid[0] * self.args.grid_size + normalized_cur_grid[1]
            if (normalized_pos_id >= self.args.grid_size ** 2) or (normalized_pos_id < 0):
                print(normalized_cur_pos)


            dx, dy, dz = episode.target_position - np.array(current_position)
            forward_stride = np.linalg.norm((dx, dy))

            # yaw angle
            next_yaw = np.arctan2(dy, dx)
            direction = 0 if forward_stride < 0.01 else modulo_radians(next_yaw - poses[i].yaw)

            # update map
            self.nav_maps[i].update_observations(poses[i])

            landmarks = self.nav_maps[i].landmark_map.get_contours()
            centroids = _convert_contours_to_centroids(landmarks)

            normalized_position = self.normalize_position(poses[i].xy, episode.map_name, self.args.map_meters)

            normalized_centroids = [self.normalize_position(Point2D(centroid[0], centroid[1]),
                                                            episode.map_name,
                                                            self.args.map_meters) for centroid in centroids]
            # normalized_centroids
            # print(landmarks, centroids)
            pred_goal_xy = np.mean(centroids, axis=0) if centroids else np.array([0, 0])

            rgb = cropclient.crop_image(episode.map_name, poses[i], (224, 224), 'rgb')
            progress = np.clip(
                1 - episode.target_position.xy.dist_to(poses[i].xy) / 100,
                0, 1)

            obs.append({
                'instruction': episode.target_description,
                'map_name': episode.map_name,
                'id': episode.id,
                'maps': self.nav_maps[i].to_array(),
                'rgb': rgb,
                'pose': poses[i],
                'goal': episode.target_position.xy,
                'direction': direction,
                'position': normalized_position,
                'cur_grid': normalized_pos_id,
                'trajectory': episode.trajectory,
                'progress': progress,
                'centroids': np.mean(normalized_centroids, axis=0) if normalized_centroids else np.array([0, 0]),
                'centroid_goal': pred_goal_xy,
                'normalized_goal': normalized_goal_xys,
                'grid_goal': normalized_goal_id
            })

            # TODO: what to use for a2c reward?
            # A3C reward. There are multiple gt end viewpoints on REVERIE.
            # if 'end_panos' in item:
            #     min_dist = np.inf
            #     for end_pano in item['end_panos']:
            #         min_dist = min(min_dist, self.shortest_distances[scan][viewpoint][end_pano])
            # else:
            #     min_dist = 0
            # obs[-1]['distance'] = min_dist

        return obs

    ############### Nav Evaluation ###############
    def _eval_item(self, gt_path, path, goal):

        scores = {}
        scores['trajectory_lengths'] = np.sum([a.dist_to(b) for a, b in zip(path[:-1], path[1:])])

        # scores['trajectory_lengths'] = scores['trajectory_lengths'] * 11.13 * 1e4
        gt_whole_lengths = np.sum([a.dist_to(b) for a, b in zip(gt_path[:-1], gt_path[1:])])
        # print(len(gt_path))
        gt_net_lengths = path[0].dist_to(goal)

        # scores['iou'] = progress[-1]  # same as compute_iou(corners[-1], gt_corners[-1]）

        scores['ne'] = path[-1].dist_to(goal)
        scores['oracle_ne'] = np.min(np.array([path[x].dist_to(goal) for x in range(len(path))]))

        scores['success'] = float(path[-1].dist_to(goal) <= self.args.success_dist)

        # scores['gp_success'] = float()
        # _center = np.mean(gt_corners[-1], axis=0)
        # _point = Point(_center)
        # _poly = Polygon(np.array(corners[-1]))
        # if not _poly.contains(_point):
        #     scores['success'] = float(0)

        # _center = np.mean(corners[-1], axis=0)
        # _point = Point(_center)
        # _poly = Polygon(np.array(gt_corners[-1]))
        # if not _poly.contains(_point):
        #     scores['success'] = float(0)

        scores['oracle_success'] = float(
            any(np.array([point.dist_to(goal) for point in path]) <= self.args.success_dist))
        scores['gt_length'] = gt_net_lengths
        scores['spl'] = scores['success'] * gt_net_lengths / max(scores['trajectory_lengths'], gt_net_lengths, 0.01)

        return scores

    def eval_metrics(self, preds):
        ''' Evaluate each agent trajectory based on how close it got to the goal location 
        the path contains [view_id, angle, vofv]'''
        # print('eval %d predictions' % (len(preds)))

        metrics = defaultdict(list)

        for k in preds.keys():
            item = preds[k]
            instr_id = item['instr_id']
            traj = [pose.xy for pose in item['trajectory']]  # x = (corners, directions)
            # corners = [np.array(x[0]) for x in item['path_corners']]  # x = (corners, directions)
            goal = [x for x in item['goal']]
            # gt_corners = [np.array(x) for x in item['gt_path_corners']]
            gt_trajs = [pose.xy for pose in item['gt_trajectory']]

            # pred_goals = [x for x in item['pred_goal']]
            # gt_goal = [x for x in item['gt_goal']]
            #
            #
            # if len(pred_goals) == 0:
            #     print('pred empty')
            # if len(gt_goal) == 0:
            #     print('gt empty')
            #
            # metrics['gp_success'].append(float(pred_goals[-1] == gt_goal[-1]))
            # metrics['oracle_gp_success'].append(
            #     float(any(np.array([pred_goal == gt_goal[-1] for pred_goal in pred_goals]))))

            # if len(traj) == 0:
            #     print('at', instr_id, traj)
            # if len(gt_trajs) == 0:
            #     print('gt', instr_id, gt_trajs)

            # for suc in item['goal_success']:
            #     metrics['oracle_goal_success'].append(suc)
            # metrics['goal_success'].append(item['goal_success'][-1])

            traj_scores = self._eval_item(gt_trajs, traj, goal)
            for k, v in traj_scores.items():
                # if k == 'iou' and traj_scores['success']:
                #     metrics[k].append(v)
                # else:
                metrics[k].append(v)

            traj = [pose.xy for pose in item['stage1_trajectory']]  # x = (corners, directions)
            if len(traj) == 0:
                traj = [item['trajectory'][0].xy]
            traj_scores = self._eval_item(gt_trajs, traj, goal)
            for k, v in traj_scores.items():
                if k == 'trajectory_lengths':
                    metrics['stage1_trajectory_lengths'].append(v)
                if k == 'success':
                    metrics['stage1_success'].append(v)
                if k == 'oracle_success':
                    metrics['stage1_oracle_success'].append(v)
                if k == 'ne':
                    metrics['stage1_ne'].append(v)
                if k == 'oracle_ne':
                    metrics['stage1_oracle_ne'].append(v)
                # # else:
                # metrics[k].append(v)

            traj = [pose.xy for pose in item['stage2_trajectory']]
            if len(traj) == 0:
                traj = [item['trajectory'][-1].xy]
            traj_scores = self._eval_item(gt_trajs, traj, goal)
            for k, v in traj_scores.items():
                if k == 'trajectory_lengths':
                    metrics['stage2_trajectory_lengths'].append(v)
                if k == 'success':
                    metrics['stage2_success'].append(v)
                if k == 'oracle_success':
                    metrics['stage2_oracle_success'].append(v)
                if k == 'ne':
                    metrics['stage2_ne'].append(v)
                if k == 'oracle_ne':
                    metrics['stage2_oracle_ne'].append(v)

            metrics['instr_id'].append(instr_id)
        avg_metrics = {
            # 'steps': np.mean(metrics['trajectory_steps']),
            'lengths': np.mean(metrics['trajectory_lengths']),
            'stage1_length': np.mean(metrics['stage1_trajectory_lengths']),
            'stage2_length': np.mean(metrics['stage2_trajectory_lengths']),
            'sr': np.mean(metrics['success']) * 100,
            'sr1': np.mean(metrics['stage1_success']) * 100,
            'sr2': np.mean(metrics['stage2_success']) * 100,
            'oracle_sr': np.mean(metrics['oracle_success']) * 100,
            'oracle_sr1': np.mean(metrics['stage1_oracle_success']) * 100,
            'oracle_sr2': np.mean(metrics['stage2_oracle_success']) * 100,
            'spl': np.mean(metrics['spl']) * 100,
            'ne': np.mean(metrics['ne']),
            'oracle_ne': np.mean(metrics['oracle_ne']),
            'stage1_ne': np.mean(metrics['stage1_ne']),
            'stage1_oracle_ne': np.mean(metrics['stage1_oracle_ne']),
            'stage2_ne': np.mean(metrics['stage2_ne']),
            'stage2_oracle_ne': np.mean(metrics['stage2_oracle_ne']),
            'gt_length': np.mean(metrics['gt_length']),
            'gp_sr': np.mean(metrics['gp_success']) * 100,
            'oracle_gp_sr': np.mean(metrics['oracle_gp_success']) * 100
            # 'oracle_goal_sr': np.mean(metrics['oracle_goal_success']),
            # 'goal_sr': np.mean(metrics['goal_success'])
            # 'oracle_pred_sr': np.mean(item['oracle_success']) * 100,
            # 'pred_sr':
            # 'iou': np.mean(metrics['iou']),
            # 'spl_short': np.mean(metrics['spl_short']) * 100,
            # 'sr_short': np.mean(metrics['success_short']) * 100,
            # 'gp_short': np.mean(metrics['gp_short']),
        }

        return avg_metrics, metrics
