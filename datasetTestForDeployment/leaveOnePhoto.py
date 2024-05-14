import os
import random

def retain_one_file(folder_path):
    """
    Retain only one file in each subfolder within the specified folder path.
    """
    # Get all subfolders in the specified folder path
    subfolders = [os.path.join(folder_path, subfolder) for subfolder in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, subfolder))]
    
    # Iterate over each subfolder
    for subfolder in subfolders:
        # Get all files in the current subfolder
        files = [os.path.join(subfolder, file) for file in os.listdir(subfolder) if os.path.isfile(os.path.join(subfolder, file))]
        
        # If there are files in the subfolder
        if len(files) > 1:
            # Randomly select one file to retain
            file_to_retain = random.choice(files)
            
            # Remove all other files
            for file in files:
                if file != file_to_retain:
                    os.remove(file)
                    print(f"Removed file: {file}")

def main():
    # Define the paths to the train and validation folders
    dataset_folder = 'dataset'
    train_folder = os.path.join(dataset_folder, 'train')
    validation_folder = os.path.join(dataset_folder, 'validation')
    
    # Retain only one file in each subfolder within train and validation folders
    retain_one_file(train_folder)
    retain_one_file(validation_folder)
    print("Completed processing all folders.")

if __name__ == "__main__":
    main()
