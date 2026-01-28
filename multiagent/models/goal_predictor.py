from collections.abc import Callable

import torch
import torch.nn as nn
from torch import Tensor


class MapEncoder(nn.Module):
    '''Encodes maps of size (240, 240, 5) into a (15 * 15 * 32) feature vector'''

    def __init__(self, map_size: int):
        super(MapEncoder, self).__init__()

        self.main = nn.Sequential(
            nn.MaxPool2d(2), nn.Conv2d(3, 32, 3, stride=1, padding=1), nn.ReLU(),
            nn.MaxPool2d(2), nn.Conv2d(32, 64, 3, stride=1, padding=1), nn.ReLU(),
            nn.MaxPool2d(2), nn.Conv2d(64, 128, 3, stride=1, padding=1), nn.ReLU(),
            nn.MaxPool2d(2), nn.Conv2d(128, 64, 3, stride=1, padding=1), nn.ReLU(),
            nn.Conv2d(64, 32, 3, stride=1, padding=1), nn.ReLU(),
            nn.Flatten()
        )

        self.out_features = (map_size // 2 ** 4) ** 2 * 32

    def forward(self, maps):
        x = self.main(maps)
        return x


class GoalPredictionHead(nn.Module):

    def __init__(self, n_map_features: int, grid_size: int):
        super(GoalPredictionHead, self).__init__()

        self.prediction_head = nn.Sequential(
            nn.Linear(n_map_features, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, grid_size ** 2),
            nn.Sigmoid(),
        )

    def forward(self, map_features):
        return self.prediction_head(map_features)


class GoalPredictor(nn.Module):

    def __init__(self, map_size: int, grid_size: int):
        super(GoalPredictor, self).__init__()

        self.map_encoder = MapEncoder(map_size)

        self.goal_prediction_head = GoalPredictionHead(self.map_encoder.out_features, grid_size)

    def forward(self, maps: Tensor):
        map_features = self.map_encoder(maps)

        pred_normalized_goal_xys = self.goal_prediction_head(map_features)

        return pred_normalized_goal_xys
