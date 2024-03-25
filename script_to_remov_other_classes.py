import os

def filter_files_in_directory(directory):
    """
    Iterate through all .txt files in the given directory (including subdirectories) and
    delete lines that do not start with '0'.
    Additionally, print the progress percentage.
    """
    # Collect all .txt file paths
    txt_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".txt"):
                txt_files.append(os.path.join(root, file))

    total_files = len(txt_files)
    for index, file_path in enumerate(txt_files, start=1):
        with open(file_path, 'r') as file:
            lines = file.readlines()
        with open(file_path, 'w') as file:
            for line in lines:
                if line.startswith('0'):
                    file.write(line)

        # Print progress
        progress = (index / total_files) * 100
        print(f"Progress: {progress:.2f}%")

# Example usage
if __name__ == '__main__':
    filter_files_in_directory('/home/pal.bentsen/D1/datasets2024v2/VISEM-Tracking-sorted-GoldLegacy-soloClass')
