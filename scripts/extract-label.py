import os
import xml.etree.ElementTree as ET

# Directory containing the InkML files
input_directory = 'mathwriting-2024-excerpt/train/'
output_directory = 'my-labels/'

# Ensure the output directory exists
os.makedirs(output_directory, exist_ok=True)

# Function to extract label and write it to a new file
def write_label_to_file(file_path):
    try:
        # Parse the XML content
        tree = ET.parse(file_path)
        root = tree.getroot()

        # Define the namespace for the InkML file (needed because the XML elements have a namespace)
        namespace = {'inkml': 'http://www.w3.org/2003/InkML'}

        # Find the annotation element with type 'label'
        label_element = root.find(".//inkml:annotation[@type='label']", namespaces=namespace)

        # Extract the file name without extension (for naming the output file)
        file_name = os.path.splitext(os.path.basename(file_path))[0]

        # Define the output file path
        output_file_path = os.path.join(output_directory, f'{file_name}.txt')

        # Extract the label's text
        if label_element is not None:
            label = label_element.text
            # Write the label to the output file
            with open(output_file_path, 'w') as output_file:
                output_file.write(f"{label}\n")
            print(f"Label written to: {output_file_path}")
        else:
            print(f"Label not found in {file_path}")

    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")

# Iterate over all files in the input directory
for file_name in os.listdir(input_directory):
    if file_name.endswith(".inkml"):
        # Get the full path of the input file
        input_file_path = os.path.join(input_directory, file_name)

        # Process the file and extract the label
        write_label_to_file(input_file_path)
