import os
import json
import pandas as pd
import numpy as np

data_output_dir = "../../data/dataset_classifier"
image_data_output_dir = os.path.join(data_output_dir, "images")
train_csv_path = os.path.join(data_output_dir, "train.csv")
val_csv_path = os.path.join(data_output_dir, "val.csv")

if not os.path.exists(data_output_dir):
    os.makedirs(data_output_dir)

aivshuman_image_dir = "../../data/aivshuman"
aivshuman_csv_file = "../../data/aivshuman/train.csv"

flickr_image_dir = "../../data/flickr/flickr30k_images"
flickr_csv_file = "../../data/flickr/results.csv"


aivshuman_data = pd.read_csv(aivshuman_csv_file)
flickr_data = pd.read_csv(flickr_csv_file, delimiter = "|")

print("AIVSHUMAN data shape:", aivshuman_data.shape)
print("FLICKR data shape:", flickr_data.shape)

print(aivshuman_data.head())
print(flickr_data.head())
print(aivshuman_data.shape)
print(flickr_data.shape)

human_data = aivshuman_data[aivshuman_data['label'] == 0]
ai_data = aivshuman_data[aivshuman_data['label'] == 1]

human_images = human_data['file_name'].tolist()
ai_images = ai_data['file_name'].tolist()

human_images = list(set(human_images))
ai_images = list(set(ai_images))

# Select random 10000 images with seed
np.random.seed(42)
human_images_final = np.random.choice(human_images, 12500, replace = False)
ai_images_final = np.random.choice(ai_images, 12500, replace = False)

#Combine the two lists
final_shutterstock_images  = list(human_images_final) + list(ai_images_final)
final_shutterstock_labels = [0] * len(final_shutterstock_images)

flickr_images = flickr_data['image_name'].tolist()
flickr_images = list(set(flickr_images))
flickr_images_final = np.random.choice(flickr_images, 25000, replace = False)
flickr_labels_final = [1] * len(flickr_images_final)

#Join paths with the image names
final_shutterstock_images = [os.path.join(aivshuman_image_dir, image) for image in final_shutterstock_images]
flickr_images_final = [os.path.join(flickr_image_dir, image) for image in flickr_images_final]
# Combine the two lists
final_images = list(final_shutterstock_images) + list(flickr_images_final)
final_labels = final_shutterstock_labels + flickr_labels_final

# Create a dataframe
df = pd.DataFrame({
    'image_name': final_images,
    'label': final_labels
})
# Shuffle the dataframe
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
# Split the dataframe into train and val dataframes
train_df = df.sample(frac=0.8, random_state=42)
val_df = df.drop(train_df.index)
# Save the dataframes to csv files
train_df.to_csv(train_csv_path, index=False)
val_df.to_csv(val_csv_path, index=False)






