import os
from PIL import Image
import cv2
import numpy as np
from sklearn.model_selection import train_test_split


def load_images_from_folder_sep(main_folder, test_size=0.2):
    images = []
    images_val = []
    labels = []
    labels_val = []
    class_names = os.listdir(main_folder)

    valid_extensions = {'.jpg', '.jpeg', '.png', '.gif'}  # Add more extensions if needed

    for class_name in class_names:
        class_path = os.path.join(main_folder, class_name)
        if os.path.isdir(class_path):
            image_files = [filename for filename in os.listdir(class_path) if
                           os.path.splitext(filename)[1].lower() in valid_extensions]
            num_images = len(image_files)

            # Calculate the number of images for training and validation
            num_validation = int(num_images * test_size)
            num_training = num_images - num_validation

            # Create labels for the class
            class_labels = [class_name] * num_images
            # Use stratified sampling to split the class into training and validation
            X_train_class, X_val_class, y_train_class, y_val_class = train_test_split(image_files, class_labels,
                                                                                      test_size=test_size,
                                                                                      random_state=42,
                                                                                      stratify=class_labels)

            # Load training images
            old = len(images)
            for filename, label in zip(X_train_class, y_train_class):
                img_path = os.path.join(class_path, filename)
                img = cv2.imread(img_path)
                if img is not None:
                    # Resize the image if needed
                    images.append(img)
                    labels.append(label)
            print("Training: ",class_name, len(images) - old)
            old = len(images_val)
            # You can use X_val_class and y_val_class similarly for the validation set
            for filename, label in zip(X_val_class, y_val_class):
                img_path = os.path.join(class_path, filename)
                img = cv2.imread(img_path)
                if img is not None:
                    # Resize the image if needed
                    images_val.append(img)
                    labels_val.append(label)
            print("Validating: ",class_name, len(images_val) - old)
    return images, labels, images_val, labels_val


def load_images_from_folder(folder_path):
    image_list = []
    labels = []
    # Loop through each folder in the specified directory
    for folder_name in os.listdir(folder_path):
        folder_path_full = os.path.join(folder_path, folder_name)
        # Check if it's a directory
        if os.path.isdir(folder_path_full):
            # Loop through each file in the folder
            old = len(image_list)
            for filename in os.listdir(folder_path_full):
                file_path = os.path.join(folder_path_full, filename)

                # Check if it's a file and has a common image extension (e.g., jpg, png)
                if os.path.isfile(file_path) and filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                    # Load the image using PIL
                    image = cv2.imread(file_path)

                    # Append the image to the list
                    image_list.append(image)
                labels.append(folder_name)
            print("Testing: ",folder_name, len(image_list) - old)
    return image_list, labels


def count_class(folder_path):
    num_class = 0
    list=[]
    for folder_name in os.listdir(folder_path):
        num_class += 1
        list.append(folder_name)
    return num_class,list