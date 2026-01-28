"""
An rgbd image client that uses cropped raster image.

Examples
--------
>>> cropclient.load_image_cache()
>>> rgb, depth = cropclient.crop_view_area('birmingham_block_0', Pose4D(350, 243, 30, np.pi/4), (500, 500))
"""
import gc
import os.path
from typing import Literal

import cv2
import numpy as np
import rasterio
import rasterio.mask
from tqdm import tqdm

from multiagent.actions import DiscreteAction
from multiagent.mapdata import GROUND_LEVEL
from multiagent.defaultpaths import ORTHO_IMAGE_DIR
from multiagent.maps.landmark_map import LandmarkMap
from multiagent.space import Pose4D, view_area_corners, Point3D, Point2D
from typing import List, Dict, Callable, Tuple

# module-wise image cache
_raster_cache = None
_rgb_cache = None
_height_cache = None  # can be converted to depth


def get_rgbd(map_name: str, pose: Pose4D, rgb_size: Tuple[int, int], depth_size: Tuple[int, int]):
    rgb = crop_image(map_name, pose, rgb_size, 'rgb')
    depth = crop_image(map_name, pose, depth_size, 'depth')

    return rgb, depth


def crop_image(map_name: str, pose: Pose4D, shape: Tuple[int, int], type: Literal['rgb', 'depth']) -> np.ndarray:
    image = (_rgb_cache if type == 'rgb' else _height_cache)[map_name]

    view_area_corners_rowcol = _compute_view_area_corners_rowcol(map_name, pose)
    view_area_corners_colrow = np.flip(view_area_corners_rowcol, axis=-1)

    img_row, img_col = shape
    img_corners_colrow = [(0, 0), (img_col - 1, 0), (img_col - 1, img_row - 1), (0, img_row - 1)]
    img_corners_colrow = np.array(img_corners_colrow, dtype=np.float32)
    img_transform = cv2.getPerspectiveTransform(view_area_corners_colrow, img_corners_colrow)
    cropped_image = cv2.warpPerspective(image, img_transform, shape)

    if type == 'depth':
        cropped_image = pose.z - cropped_image
        cropped_image = cropped_image[..., np.newaxis]

    return cropped_image



def _compute_view_area_corners_rowcol(map_name: str, pose: Pose4D):
    """Returns the [front-left, front-right, back-right, back-left] corners of
    the view area in (row, col) order
    """

    raster = _raster_cache[map_name]

    view_area_corners_rowcol = [raster.index(x, y) for x, y in view_area_corners(pose, GROUND_LEVEL[map_name])]

    return np.array(view_area_corners_rowcol, dtype=np.float32)


def load_image_cache(image_dir=ORTHO_IMAGE_DIR, alt_env: Literal['', 'flood', 'ground_fissure'] = ''):
    if alt_env:
        image_dir = image_dir / alt_env

    global _raster_cache, _rgb_cache, _height_cache

    if _raster_cache is None:
        _raster_cache = {
            raster_path.stem: rasterio.open(raster_path)
            for raster_path in image_dir.glob("*.tif")
        }

    if _rgb_cache is None:
        _rgb_cache = {
            rgb_path.stem: cv2.cvtColor(cv2.imread(str(rgb_path)), cv2.COLOR_BGR2RGB)
            for rgb_path in tqdm(image_dir.glob("*.png"), desc="reading rgb data from disk", leave=False)
        }

    # if _height_cache is None:
    #     _height_cache = {
    #         map_name: raster.read(1)  # read first channel (1-based index)
    #         for map_name, raster in tqdm(_raster_cache.items(), desc="reading depth data from disk", leave=False)
    #     }


def clear_image_cache():
    global _raster_cache, _rgb_cache, _height_cache

    if _raster_cache is not None:
        for dataset in _raster_cache:
            dataset.close()

    _raster_cache = _rgb_cache = _height_cache = None
    gc.collect()


# def save_actions(map_name: str, actions, teacher_trajectory, description: str, save_path: str):
#     image = _rgb_cache[map_name].copy()
#     # for pose in poses:
#     #     view_area_corners_rowcol = _compute_view_area_corners_rowcol(map_name, pose)
#     #     view_area_corners_colrow = np.flip(view_area_corners_rowcol, axis=-1)
#
#     #     view_area_corners_colrow_int = np.int32(view_area_corners_colrow)
#     #     outline_color = (0, 255, 0)  # 绿色
#     #     thickness = 2
#     #     image_with_outline = cv2.polylines(image, [view_area_corners_colrow_int], isClosed=True, color=outline_color, thickness=thickness)
#     last = None
#     thickness = 2
#     for action, pose in zip(actions, teacher_trajectory):
#         view_area_corners_rowcol = _compute_view_area_corners_rowcol(map_name, pose)
#         view_area_corners_colrow = np.flip(view_area_corners_rowcol, axis=-1)
#
#         view_area_corners_colrow_int = np.int32(view_area_corners_colrow)
#         # print(view_area_corners_colrow_int.shape)
#         center = np.mean(view_area_corners_colrow_int, axis=0)
#         # print(center.shape)
#         center = (int(center[0]), int(center[1]))
#
#         if last != None:
#             cv2.line(image, last, center, (0, 0, 255), thickness)
#             # dst = Point3D(pose.x, pose.y, pose.z)
#             # action = lookahead_discrete_action(last_pose, [dst])
#         cv2.putText(image, list(DiscreteAction.__members__.keys())[action],
#                     center,
#                     cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1, cv2.LINE_AA)
#         last = center
#
#     view_area_corners_rowcol = _compute_view_area_corners_rowcol(map_name, teacher_trajectory[0])
#     view_area_corners_colrow = np.flip(view_area_corners_rowcol, axis=-1)
#     view_area_corners_colrow_int = np.int32(view_area_corners_colrow)
#
#     outline_color = (255, 0, 0)  # 绿色
#
#     image_with_outline = cv2.polylines(image, [view_area_corners_colrow_int], isClosed=True, color=outline_color,
#                                        thickness=thickness)
#     outline_color = (0, 0, 255)  # 绿色
#
#     view_area_corners_rowcol = _compute_view_area_corners_rowcol(map_name, teacher_trajectory[-1])
#     view_area_corners_colrow = np.flip(view_area_corners_rowcol, axis=-1)
#     view_area_corners_colrow_int = np.int32(view_area_corners_colrow)
#     image_with_outline = cv2.polylines(image_with_outline, [view_area_corners_colrow_int], isClosed=True,
#                                        color=outline_color, thickness=thickness)
#
#     cv2.putText(image_with_outline, description,
#                 (50, 50),
#                 cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1, cv2.LINE_AA)
#
#     # 2. 保存带边界的原图
#     # output_path = "output_image_with_outline.png"
#     cv2.imwrite(save_path, image_with_outline)
#
#
# def save_logs(map_name: str, stage1_poses, stage2_poses, teacher_trajectory, description: str, save_path: str, centroid: np.array):
#     image = _rgb_cache[map_name].copy()
    for pose in stage1_poses:
        view_area_corners_rowcol = _compute_view_area_corners_rowcol(map_name, pose)
        view_area_corners_colrow = np.flip(view_area_corners_rowcol, axis=-1)

        view_area_corners_colrow_int = np.int32(view_area_corners_colrow)
        outline_color = (0, 255, 0)  # 绿色
        thickness = 2
        image = cv2.polylines(image, [view_area_corners_colrow_int], isClosed=True,
                                           color=outline_color, thickness=thickness)
#     for pose in stage2_poses:
#         view_area_corners_rowcol = _compute_view_area_corners_rowcol(map_name, pose)
#         view_area_corners_colrow = np.flip(view_area_corners_rowcol, axis=-1)
#
#         view_area_corners_colrow_int = np.int32(view_area_corners_colrow)
#         outline_color = (255, 0, 0)  # 绿色
#         thickness = 2
#         image = cv2.polylines(image, [view_area_corners_colrow_int], isClosed=True,
#                                            color=outline_color, thickness=thickness)
#
#     for i, pose in enumerate(teacher_trajectory):
#         view_area_corners_rowcol = _compute_view_area_corners_rowcol(map_name, pose)
#         view_area_corners_colrow = np.flip(view_area_corners_rowcol, axis=-1)
#
#         view_area_corners_colrow_int = np.int32(view_area_corners_colrow)
#         # print(view_area_corners_colrow_int.shape)
#         center = np.mean(view_area_corners_colrow_int, axis=0)
#         # print(center.shape)
#         center = (int(center[0]), int(center[1]))
#
#         if i != 0:
#             cv2.line(image, last, center, (0, 0, 255), thickness)
#             dst = Point3D(pose.x, pose.y, pose.z)
#             # action = lookahead_discrete_action(last_pose, [dst])
#             # cv2.putText(image, str(action),
#             #             last,
#             #             cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1, cv2.LINE_AA)
#         last = center
#         last_pose = pose
#         outline_color = (0, 0, 0)  # 绿色
#         thickness = 2
#         if i == 0:
#             image = cv2.polylines(image, [view_area_corners_colrow_int], isClosed=True,
#                                                color=outline_color, thickness=thickness)
#         outline_color = (0, 0, 255)  # 绿色
#         if i == len(teacher_trajectory) - 1:
#             image = cv2.polylines(image, [view_area_corners_colrow_int], isClosed=True,
#                                                color=outline_color, thickness=thickness)
#
#         cv2.putText(image, description,
#                     (50, 50),
#                     cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1, cv2.LINE_AA)
#
#     # landmarks = LandmarkMap._search_landmarks_by_name()
#     # 2. 保存带边界的原图
#     # output_path = "output_image_with_outline.png"
#     outline_color = (0, 0, 255)
#     point = [_raster_cache[map_name].index(centroid[0], centroid[1])[1], _raster_cache[map_name].index(centroid[0], centroid[1])[0]]
#     cv2.circle(image, point, 20, outline_color, 5)
#
#     # contours = []
#     # for lm in landmarks:
#     #     contour = [[_raster_cache[map_name].index(x, y)[1], _raster_cache[map_name].index(x, y)[0]] for (x, y) in lm]
#     #
#     #     # print(type(contour[0]))
#     #     # print(contour)
#     #     contour = np.array(contour)
#     #     # print(contour)
#     #     # contours.append(contour)
#     #     # print(contour.shape)
#     # # if len(contours) > 0:
#     # #     print(len(contours))
#     #     cv2.drawContours(image, [contour], -1, (255, 255, 255), 3)
#     cv2.imwrite(save_path, image)

def save_grids(map_name: str, stage1_traj, stage2_traj, teacher_trajectory, landmarks: List[List[Point2D]],
               description_landmarks: List[str], save_path: str, goals):

    os.makedirs(save_path, exist_ok=True)

    image = _rgb_cache[map_name].copy()
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)



    gray_value = 128
    alpha = 0.5

    overlay = np.full_like(image, (gray_value, gray_value, gray_value))

    image = cv2.addWeighted(image, 1 - alpha, overlay, alpha, 0)
    image_ori = image.copy()
    view_area_corners_rowcol = _compute_view_area_corners_rowcol(map_name, teacher_trajectory[-1])
    view_area_corners_colrow = np.flip(view_area_corners_rowcol, axis=-1)

    view_area_corners_colrow_int = np.int32(view_area_corners_colrow)
    center = np.mean(view_area_corners_colrow_int, axis=0)
    # print(center.shape)
    center = (int(center[0]), int(center[1]))
    outline_color = (0, 255, 0)  # 绿色
    # thickness = 2
    #
    # image = cv2.polylines(image, [view_area_corners_colrow_int], isClosed=True,
    #                       color=outline_color, thickness=thickness)
    cv2.circle(image, center, 40, outline_color, 15)



    pred_image = image.copy()
    for i, goal in enumerate(goals):
        center = [_raster_cache[map_name].index(goal[0], goal[1])[1], _raster_cache[map_name].index(goal[0], goal[1])[0]]
        # center = (int(goal[0]), int(goal[1]))
        # if i != 0:
        cv2.circle(pred_image, center, 40, outline_color, 15)
        # dst = Point3D(pose.x, pose.y, pose.z)
        # action = lookahead_discrete_action(last_pose, [dst])
        cv2.putText(pred_image, str(i),
                    center,
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite(os.path.join(save_path, 'pred.png'), pred_image)

    # print(len(goals), len(stage1_traj))



    for i, pose in enumerate(stage1_traj):

        cur_image = image.copy()

        view_area_corners_rowcol = _compute_view_area_corners_rowcol(map_name, pose)
        view_area_corners_colrow = np.flip(view_area_corners_rowcol, axis=-1)

        view_area_corners_colrow_int = np.int32(view_area_corners_colrow)
        outline_color = (0, 255, 255)  # 绿色
        thickness = 20
        # image = cv2.polylines(image, [view_area_corners_colrow_int], isClosed=True,
        #                       color=outline_color, thickness=thickness)
        center = np.mean(view_area_corners_colrow_int, axis=0)
        # print(center.shape)
        center = (int(center[0]), int(center[1]))

        # cur_view = cv2.cvtColor(crop_image(map_name, pose, (224, 224), 'rgb'),cv2.COLOR_BGR2RGB)
        # cv2.imwrite(os.path.join(save_path, 'view_%d.png'%i), cur_view)



        if i != 0:
            cv2.line(image, last, center, (0, 255, 255), thickness)
            cv2.line(cur_image, last, center, (0, 0, 255), thickness)
            g = [_raster_cache[map_name].index(goals[i - 1][0], goals[i - 1][1])[1],
                      _raster_cache[map_name].index(goals[i - 1][0], goals[i - 1][1])[0]]
            cv2.circle(cur_image, g, 40, outline_color, 15)
            # cv2.imwrite(os.path.join(save_path, 'stage1_%d.png' % i), cur_image)

            dst = Point3D(pose.x, pose.y, pose.z)
            # action = lookahead_discrete_action(last_pose, [dst])
            # cv2.putText(image, str(action),
            #             last,
            #             cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1, cv2.LINE_AA)
        last = center

    for i, pose in enumerate(stage2_traj):
        view_area_corners_rowcol = _compute_view_area_corners_rowcol(map_name, pose)
        view_area_corners_colrow = np.flip(view_area_corners_rowcol, axis=-1)

        view_area_corners_colrow_int = np.int32(view_area_corners_colrow)
        outline_color = (0, 255, 0)  # 绿色
        thickness = 20
        # image = cv2.polylines(image, [view_area_corners_colrow_int], isClosed=True,
        #                       color=outline_color, thickness=thickness)
        center = np.mean(view_area_corners_colrow_int, axis=0)
        # print(center.shape)
        center = (int(center[0]), int(center[1]))

        # cur_view = cv2.cvtColor(crop_image(map_name, pose, (224, 224), 'rgb'),cv2.COLOR_BGR2RGB)
        # cv2.imwrite(os.path.join(save_path, 'view_%d.png'%(i + len(stage1_traj))), cur_view)

        if i != 0:
            cv2.line(image, last, center, (0, 0, 255), thickness)
            dst = Point3D(pose.x, pose.y, pose.z)
            # action = lookahead_discrete_action(last_pose, [dst])
            # cv2.putText(image, str(action),
            #             last,
            #             cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1, cv2.LINE_AA)
        last = center
    # cv2.imwrite(os.path.join(save_path, 'stage2.png'), image)

    for i, pose in enumerate(teacher_trajectory):
        view_area_corners_rowcol = _compute_view_area_corners_rowcol(map_name, pose)
        view_area_corners_colrow = np.flip(view_area_corners_rowcol, axis=-1)

        view_area_corners_colrow_int = np.int32(view_area_corners_colrow)
        outline_color = (0, 255, 0)  # 绿色
        thickness = 20
        # image = cv2.polylines(image, [view_area_corners_colrow_int], isClosed=True,
        #                       color=outline_color, thickness=thickness)
        center = np.mean(view_area_corners_colrow_int, axis=0)
        # print(center.shape)
        center = (int(center[0]), int(center[1]))

        if i != 0:
            cv2.line(image, last, center, (0, 255, 0), thickness)
            dst = Point3D(pose.x, pose.y, pose.z)
            # action = lookahead_discrete_action(last_pose, [dst])
            # cv2.putText(image, str(action),
            #             last,
            #             cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1, cv2.LINE_AA)
        last = center

    lm_image = image_ori.copy()
    for lm, lm_name in zip(landmarks, description_landmarks):
        contour = [[_raster_cache[map_name].index(x, y)[1], _raster_cache[map_name].index(x, y)[0]] for (x, y) in lm]

        # print(type(contour[0]))
        # print(contour)
        contour = np.array(contour)
        center = np.mean(contour, axis=0, dtype=np.int32)
        # print(center)
        # print(contour)
        # contours.append(contour)
        # print(contour.shape)
    # if len(contours) > 0:
    #     print(len(contours))
        cv2.drawContours(image, [contour], -1, (255, 255, 255), 20)

        font_face = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 3
        thickness = 3

        (text_width, text_height), baseline = cv2.getTextSize(
            lm_name, font_face, font_scale, thickness
        )

        # 计算背景框坐标
        x, y = center
        padding = 1
        bg_org1 = (x - padding, y + padding)  # 左上角
        bg_org2 = (x + text_width + padding, y - text_height - padding - baseline)  # 右下角

        bg_color = (0, 0, 255)
        # 绘制背景矩形
        # cv2.rectangle(image, bg_org1, bg_org2, bg_color, -1)  # -1 表示填充
        #
        # cv2.putText(image, lm_name, center, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
    # cv2.imwrite(os.path.join(save_path, 'landmark.png'), lm_image)

    cv2.imwrite(os.path.join(save_path, 'full.png'), image)

def save_contour(map_name: str, save_path: str, landmarks: List[List[Point2D]], description_landmarks: List[str]):
    image = _rgb_cache[map_name].copy()

    # landmarks = LandmarkMap._search_landmarks_by_name()
    # 2. 保存带边界的原图
    # output_path = "output_image_with_outline.png"
    contours = []
    for lm, lm_name in zip(landmarks, description_landmarks):
        contour = [[_raster_cache[map_name].index(x, y)[1], _raster_cache[map_name].index(x, y)[0]] for (x, y) in lm]

        # print(type(contour[0]))
        # print(contour)
        contour = np.array(contour)
        center = np.mean(contour, axis=0, dtype=np.int)
        print(center)
        # print(contour)
        # contours.append(contour)
        # print(contour.shape)
    # if len(contours) > 0:
    #     print(len(contours))
        cv2.drawContours(image, [contour], -1, (255, 255, 255), 5)

        font_face = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 3
        thickness = 3

        (text_width, text_height), baseline = cv2.getTextSize(
            lm_name, font_face, font_scale, thickness
        )

        # 计算背景框坐标
        x, y = center
        padding = 1
        bg_org1 = (x - padding, y + padding)  # 左上角
        bg_org2 = (x + text_width + padding, y - text_height - padding - baseline)  # 右下角

        bg_color = (0, 0, 255)
        # 绘制背景矩形
        cv2.rectangle(image, bg_org1, bg_org2, bg_color, -1)  # -1 表示填充

        cv2.putText(image, lm_name, center, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
    cv2.imwrite(save_path, image)
