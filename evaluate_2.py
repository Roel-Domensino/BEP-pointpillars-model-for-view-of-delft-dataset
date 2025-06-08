from vod.evaluation import Evaluation
import os

# When the instance is created, the label locations are required.
test_annotation_file= os.path.join('/home/student2/Documents/Datasets/View_of_Delft_dataset_PUBLIC/view_of_delft_PUBLIC/radar','label_2_edited')
evaluation = Evaluation(test_annotation_file)

# Using the evaluate method, the model can be evaluated on the detection labels.
results = evaluation.evaluate(
    result_path=os.path.join('example_set', 'detection'),
    current_class=[0, 1, 2])

print("Results: \n"
      f"Entire annotated area: \n"
      f"Car: {results['entire_area']['Car_3d_all']} \n"
      f"Pedestrian: {results['entire_area']['Pedestrian_3d_all']} \n"
      f"Cyclist: {results['entire_area']['Cyclist_3d_all']} \n"
      f"mAP: {(results['entire_area']['Car_3d_all'] + results['entire_area']['Pedestrian_3d_all'] + results['entire_area']['Cyclist_3d_all']) / 3} \n"
      f"Driving corridor area: \n"
      f"Car: {results['roi']['Car_3d_all']} \n"
      f"Pedestrian: {results['roi']['Pedestrian_3d_all']} \n"
      f"Cyclist: {results['roi']['Cyclist_3d_all']} \n"
      f"mAP: {(results['roi']['Car_3d_all'] + results['roi']['Pedestrian_3d_all'] + results['roi']['Cyclist_3d_all']) / 3} \n"
      )