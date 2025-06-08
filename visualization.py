import os
import numpy as np
from vod.frame import FrameDataLoader
from vod.visualization import Visualization2D

from vod.configuration import KittiLocations
from vod.frame import FrameDataLoader
from vod.frame import homogeneous_transformation, FrameTransformMatrix

kitti_locations = KittiLocations(root_dir="/home/student2/Documents/Datasets/View_of_Delft_dataset_PUBLIC/view_of_delft_PUBLIC",
                                output_dir="example_output",
                                pred_dir="/home/student2/Documents/Datasets/View_of_Delft_dataset_PUBLIC/view_of_delft_PUBLIC/predictions_6000")

frame_data = FrameDataLoader(kitti_locations=kitti_locations,
                             frame_number="01201")


transforms = FrameTransformMatrix(frame_data)

#frame_path= os.path.join('/home/student2/Documents/Datasets/View_of_Delft_dataset_PUBLIC/view_of_delft_PUBLIC/radar','label_2_edited')
#frame_data = FrameDataLoader(kitti_locations=frame_path,
#                             frame_number="01201")
vis2d = Visualization2D(frame_data)

vis2d.draw_plot(show_radar=True, show_gt=False, show_pred=True)
#vis2d.draw_plot( show_pred=True)