import os

# Directory containing your text files
directory = '/home/pal.bentsen/D1/frames_without_val_participants'
files = [f for f in os.listdir(directory) if f.endswith('.txt')]

total_files = len(files)
processed_count = 0

for filename in files:
    filepath = os.path.join(directory, filename)
    
    # Read in all lines from the file
    with open(filepath, 'r') as file:
        lines = file.readlines()

    # Process each line
    processed_lines = []
    for line in lines:
        parts = line.strip().split(' ')
        # Remove the last element (confidence score)
        new_line = ' '.join(parts[:-1]) + '\n'
        processed_lines.append(new_line)

    # Write the processed lines back to the file
    with open(filepath, 'w') as file:
        file.writelines(processed_lines)
    
    # Update progress
    processed_count += 1
    progress = (processed_count / total_files) * 100
    print(f"Progress: {progress:.2f}%", end='\r')

print("\nProcessing complete.")

