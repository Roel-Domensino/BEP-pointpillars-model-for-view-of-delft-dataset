import os
import shutil

# Specify the folder path and the target folder where the filtered files will be copied
source_folder = "/home/student2/Documents/Datasets/View_of_Delft_dataset_PUBLIC/view_of_delft_PUBLIC/radar/training/label_2"  # Replace with your source folder path
target_folder = "/home/student2/Documents/Datasets/View_of_Delft_dataset_PUBLIC/view_of_delft_PUBLIC/radar/training/label_2_edited"  # Replace with your target folder path

# Ensure the target folder exists, if not, create it
if not os.path.exists(target_folder):
    os.makedirs(target_folder)

# Class names to keep
valid_classes = {'Car', 'Pedestrian', 'bicycle','Cyclist', 'moped_scooter','motor','ride_other','rider','bicycle_rack'}

def filter_lines(file_path):
    """
    Reads a file, filters out lines that do not contain a valid class, and returns the filtered lines.
    """
    filtered_lines = []
    with open(file_path, 'r') as file:
        for line in file:
            # Extract the class name from the line (the first word)
            class_name = line.split()[0]
            if class_name in valid_classes:
                filtered_lines.append(line)
    return filtered_lines

def copy_and_filter_files(source_folder, target_folder):
    """
    Copies all text files from the source folder to the target folder, filtering out lines based on class name.
    """
    for filename in os.listdir(source_folder):
        if filename.endswith('.txt'):  # Process only .txt files
            source_file_path = os.path.join(source_folder, filename)
            target_file_path = os.path.join(target_folder, filename)

            # Filter the lines of the current file
            filtered_lines = filter_lines(source_file_path)

            # Write the filtered lines to the new file in the target folder
            with open(target_file_path, 'w') as target_file:
                target_file.writelines(filtered_lines)

            print(f"Processed {filename}, filtered lines written to {target_file_path}")

# Run the function to copy and filter the files
copy_and_filter_files(source_folder, target_folder)