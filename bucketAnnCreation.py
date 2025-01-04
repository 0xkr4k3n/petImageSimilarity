import boto3
import os
import torch
from PIL import Image
from torchvision import models, transforms
import torch.nn as nn
from annoy import AnnoyIndex

# AWS credentials
access_key = '404_key_not_found'
secret_key = '404_key_not_found'
region = 'us-east-1'

# Bus
source_bucket_name = 'petpulse'
target_bucket_name = 'similarityfeatureanns'

# Create a new S3 client for the source bucket
s3_client_source = boto3.client('s3', aws_access_key_id=access_key, aws_secret_access_key=secret_key, region_name=region)

# List objects in the source bucket
response = s3_client_source.list_objects(Bucket=source_bucket_name)

# Filter and select only .jpg, .jpeg, .png images
images = [obj['Key'] for obj in response.get('Contents', []) if obj['Key'].lower().endswith(('.jpg', '.jpeg', '.png'))]

# Use ResNet50 model for feature extraction
weights = models.resnet50(pretrained=True)
model = models.resnet50(weights=weights)
model.fc = nn.Identity()
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Increase dimensionality of feature vectors to 2048 for ResNet50
annoy_index = AnnoyIndex(2048, 'angular')

for i, image_name in enumerate(images):
    # Download the image from the source bucket
    file_name = f'./downloaded_images/{image_name}'  # Temporary file to store the downloaded image
    s3_client_source.download_file(source_bucket_name, image_name, file_name)
    
    # Process the downloaded image
    image = Image.open(file_name)
    input_tensor = transform(image).unsqueeze(0)

    if input_tensor.size()[1] == 3:
        output_tensor = model(input_tensor)
        annoy_index.add_item(i, output_tensor[0])
        
        if i % 100 == 0:
            print(f'Processed {i} images.')

annoy_index.build(100)

# Save the Annoy index file
index_file_name = 'dog_index_moreacc.ann'
annoy_index.save(index_file_name)

# Create a new S3 client for the target bucket
s3_client_target = boto3.client('s3', aws_access_key_id='404_key_not_found', aws_secret_access_key='404_key_not_found', region_name=region)

# Upload the Annoy index file to the target bucket
with open(index_file_name, 'rb') as file:
    s3_client_target.upload_fileobj(file, target_bucket_name, index_file_name)

# Clean up temporary files
os.remove(file_name)
os.remove(index_file_name)

print(f'Annoy index saved and uploaded to bucket: {target_bucket_name}')
