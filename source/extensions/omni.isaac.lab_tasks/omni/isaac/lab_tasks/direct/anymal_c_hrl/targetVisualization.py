import torch

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.markers import VisualizationMarkers, VisualizationMarkersCfg
from omni.isaac.lab.sim import SimulationContext
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from omni.isaac.lab.utils.math import quat_from_angle_axis

class targetVisualization:
    def __init__(self, scale=1.0, num_envs=1):
        self.markers = self.define_markers(scale)
        self.num_envs = num_envs
        self.target_pos_w = None
        self.target_rot_w =  self.get_marker_orientation()
        self.indices = torch.zeros(num_envs)
        
    def set_marker_position(self, target_pos_r, root_pos_w) -> torch.Tensor:
        # return the target position in world coordinates   
        self.target_pos_w =  root_pos_w + target_pos_r
        return self.target_pos_w

    def get_marker_orientation(self) -> torch.Tensor:
        # return the target orientation in world coordinates all zeros
        rot = torch.zeros(self.num_envs)
        target_rot_w =  quat_from_angle_axis(rot, torch.tensor([0.0, 0.0, 1.0]))
        return target_rot_w

    def define_markers(self,scale) -> VisualizationMarkers:
        """Define markers with various different shapes."""
        marker_cfg = VisualizationMarkersCfg(
            prim_path="/Visuals/myMarkers",
            markers={
                "sphere_untouched": sim_utils.SphereCfg(
                    radius=scale,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
                ),
                "sphere_touched": sim_utils.SphereCfg(
                    radius=scale,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
                ),
            },
        )
        return VisualizationMarkers(marker_cfg)
    def check_marker_touched(self, distance, threshold):
        # check if the marker is touched
        self.indices = torch.where(distance < threshold, 1, 0)
    
    def reset_indices(self, reset_ids):
        # reset the indices
        self.indices[reset_ids] = 0
    
    def visualize(self, marker_indices = None):
        self.markers.visualize(self.target_pos_w, self.target_rot_w, marker_indices = self.indices)

# class targetVisualization:
#     def __init__(self, target_pos_r, root_pos_w, scale=1.0, num_envs=1):
#         self.markers = self.define_markers(scale)
#         self.target_pos_r = target_pos_r
#         self.root_pos_w = root_pos_w
#         self.target_pos_w = self.get_marker_position()
#         self.target_rot_w =  self.get_marker_orientation()
#         self.indices = torch.arange(num_envs)
# 
#     def get_marker_position(self) -> torch.Tensor:
#         # return the target position in world coordinates   
#         target_pos_w = self.root_pos_w + self.target_pos_r
#         return target_pos_w
# 
#     def get_marker_orientation(self) -> torch.Tensor:
#         # return the target orientation in world coordinates all zeros
#         rot = torch.tensor([0.0])
#         target_rot_w =  quat_from_angle_axis(rot, torch.tensor([0.0, 0.0, 1.0]))
#         return target_rot_w
# 
#     def define_markers(self,scale) -> VisualizationMarkers:
#         """Define markers with various different shapes."""
#         marker_cfg = VisualizationMarkersCfg(
#             prim_path="/Visuals/myMarkers",
#             markers={
#                 "sphere": sim_utils.SphereCfg(
#                     radius=scale,
#                     visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
#                 ),
#             },
#         )
#         return VisualizationMarkers(marker_cfg)
# 
#     def visualize(self):
#         self.markers.visualize(self.target_pos_w, self.target_rot_w, marker_indices = self.indices)