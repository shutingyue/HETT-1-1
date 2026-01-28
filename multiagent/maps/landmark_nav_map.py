from typing import Optional

import numpy as np

# from gsamllavanav.observation import cropclient
# from gsamllavanav.defaultpaths import GSAM_MAPS_DIR
from multiagent.space import Point2D, Pose4D
from multiagent.dataset.episode import Episode

from .map import Map
from .tracking_map import TrackingMap
from .landmark_map import LandmarkMap
from typing import List, Dict, Callable, Tuple

class LandmarkNavMap(Map):
    def __init__(
            self,
            map_name: str,
            map_shape: Tuple[int, int],
            map_pixels_per_meter: float,
            landmark_names: List[str],
            # target_name: str, surroundings_names: List[str],
    ):
        super().__init__(map_name, map_shape, map_pixels_per_meter)

        self.tracking_map = TrackingMap(map_name, map_shape, map_pixels_per_meter)
        self.landmark_map = LandmarkMap(map_name, map_shape, map_pixels_per_meter, landmark_names)

    def update_observations(
            self,
            camera_pose: Pose4D,
    ):
        self.tracking_map.mark_current_view_area(camera_pose)

    def to_array(self, dtype=np.float32) -> np.ndarray:
        return np.concatenate([
            self.tracking_map.to_array(dtype),
            self.landmark_map.to_array(dtype),
        ])

    @classmethod
    def generate_maps_for_a_trajectory(
            cls,
            episode: Episode,
            map_shape: Tuple[int, int],
            pixels_per_meter: float,
            trajectory: List[Pose4D],
    ):
        # tracking map
        tracking_map = TrackingMap(episode.map_name, map_shape, pixels_per_meter)
        tracking_maps = np.stack([tracking_map.mark_current_view_area(pose).to_array() for pose in trajectory])
        # print(tracking_maps.shape)
        tracking_maps = tracking_maps[-1]
        assert tracking_maps.shape == (2, *map_shape)

        # landmark maps
        landmark_map = LandmarkMap(episode.map_name, map_shape, pixels_per_meter,
                                   episode.target_processed_description.landmarks)
        landmark_maps = landmark_map.to_array()
        assert landmark_maps.shape == (1, *map_shape)

        episode_maps = np.concatenate((tracking_maps, landmark_maps), axis=0)
        assert episode_maps.shape == (3, *map_shape)
        return episode_maps
        # # tracking map
        # tracking_map = TrackingMap(episode.map_name, map_shape, pixels_per_meter)
        # tracking_maps = np.stack([tracking_map.mark_current_view_area(pose).to_array() for pose in trajectory])
        # assert tracking_maps.shape == (len(trajectory), 2, *map_shape)
        #
        # # landmark maps
        # landmark_map = LandmarkMap(episode.map_name, map_shape, pixels_per_meter, episode.target_processed_description.landmarks)
        # landmark_maps = np.tile(landmark_map.to_array(), (len(trajectory), 1, 1, 1))
        # assert landmark_maps.shape == (len(trajectory), 1, *map_shape)
        #
        #
        #
        # episode_maps = np.concatenate((tracking_maps, landmark_maps), axis=1)
        # assert episode_maps.shape == (len(trajectory), 3, *map_shape)
        # return episode_maps

    @classmethod
    def generate_maps_for_an_episode(
            cls,
            episode: Episode,
            map_shape: Tuple[int, int],
            pixels_per_meter: float,
            update_interval: int
    ):
        trajectory = episode.sample_trajectory(update_interval, -1)

        # tracking map
        tracking_map = TrackingMap(episode.map_name, map_shape, pixels_per_meter)
        tracking_maps = np.stack([tracking_map.mark_current_view_area(pose).to_array() for pose in trajectory])
        assert tracking_maps.shape == (len(trajectory), 2, *map_shape)

        # landmark maps
        landmark_map = LandmarkMap(episode.map_name, map_shape, pixels_per_meter,
                                   episode.target_processed_description.landmarks)
        landmark_maps = np.tile(landmark_map.to_array(), (len(trajectory), 1, 1, 1))
        assert landmark_maps.shape == (len(trajectory), 1, *map_shape)


        episode_maps = np.concatenate((tracking_maps, landmark_maps), axis=1)
        assert episode_maps.shape == (len(trajectory), 3, *map_shape)
        return episode_maps

    @classmethod
    def from_array(
            cls,
            map_name: str,
            map_shape: Tuple[int, int],
            map_pixels_per_meter: float,
            landmark_names: List[str],
            target_name: str,
            object_names: List[str],
            map_data: np.ndarray,
    ):
        nav_map = LandmarkNavMap(map_name, map_shape, map_pixels_per_meter, landmark_names, target_name, object_names)
        nav_map.tracking_map.current_view_area = map_data[0].astype(np.uint8)

        return nav_map

    def plot(
            self,
            goal_description: str,
            predicted_goal: Point2D,
            true_goal: Point2D,
            show=False,
    ):
        import cv2

        predicted_goal_map = cv2.circle(
            img=np.zeros(self.shape, dtype=np.float32),
            center=self.to_row_col(predicted_goal)[::-1],
            radius=4, color=1, thickness=-1
        )

        true_goal_map = cv2.circle(
            img=np.zeros(self.shape, dtype=np.float32),
            center=self.to_row_col(true_goal)[::-1],
            radius=4, color=1, thickness=-1
        )

        titles = ['current view area', 'explored area', 'landmarks', 'target', 'surroundings', 'predicted goal',
                  'true goal']
        maps = np.concatenate([self.to_array(), np.stack([predicted_goal_map, true_goal_map])])

        import matplotlib.pyplot as plt
        from PIL import Image

        fig, axs = plt.subplots(nrows=1, ncols=7, figsize=(35, 5), subplot_kw={'xticks': [], 'yticks': []})
        fig.suptitle(f"{self.name}: {goal_description}")

        for ax, title, m in zip(axs, titles, maps):
            ax.imshow(m, cmap='viridis')
            ax.set_title(title)

        plt.tight_layout()
        fig.canvas.draw()

        if show:
            plt.show()

        plot_img = Image.frombytes('RGB', fig.canvas.get_width_height(), fig.canvas.tostring_rgb())

        plt.close(fig)

        return plot_img
