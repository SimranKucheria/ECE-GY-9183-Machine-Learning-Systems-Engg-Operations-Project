import os
from PIL import Image
import pandas as pd

csv_path = "/Users/anshsarkar/NYU/Spring 2025/ECE-GY-9183-MLOPS/Project/ECE-GY-9183-Machine-Learning-Systems-Engg-Operations-Project/notebooks/data/dataset_classifier/train.csv"
image_column = "image_name"

df = pd.read_csv(csv_path)
base_dir = os.path.dirname(csv_path)

# Track validation results
results = {
    'missing_files': [],
    'invalid_channels': [],
    'all_valid': True
}

for image_name in df[image_column]:
    img_path = os.path.join(base_dir, image_name)
    
    if not os.path.exists(img_path):
        results['missing_files'].append(image_name)
        results['all_valid'] = False
        continue
        
    with Image.open(img_path) as img:
        if len(img.getbands()) != 3:
            results['invalid_channels'].append(image_name)
            results['all_valid'] = False

# Print summary
print(f"Validation complete:")
print(f"Total images checked: {len(df)}")
print(f"Missing files: {len(results['missing_files'])}")
print(f"Images with ≠3 channels: {len(results['invalid_channels'])}")
print(f"All images valid: {results['all_valid']}")
print(results['invalid_channels'])
