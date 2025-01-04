import boto3
import time
import os
from PIL import Image
from torchvision import models, transforms
import torch.nn as nn
from annoy import AnnoyIndex  # Import the AnnoyIndex class

# Define AWS credentials and S3 bucket key
aws_access_key_id = '404_key_not_found'
aws_secret_access_key = '404_key_not_found'
bucket_name = 'petspulseimages'
annoy_bucket = 'similarityfeatureanns'
annoy_key = 'dog_index_moreacc.ann'


# Create an S3 client
s3_client = boto3.client('s3', aws_access_key_id=aws_access_key_id, aws_secret_access_key=aws_secret_access_key, region_name='us-east-1')

# Keep track of processed keys and timestamps to avoid duplicates
processed_objects = {}

# Function to fetch all existing objects in the bucket
def fetch_existing_objects():
    response = s3_client.list_objects_v2(Bucket=bucket_name)
    if 'Contents' in response:
        for obj in response['Contents']:
            key = obj['Key']
            timestamp = obj['LastModified'].replace(tzinfo=None)  # Remove timezone for comparison
            processed_objects[key] = timestamp

# Define the function to process S3 events for new elements
def process_s3_events():
    while True:
        response = s3_client.list_objects_v2(Bucket=bucket_name)
        if 'Contents' in response:
            for obj in response['Contents']:
                key = obj['Key']
                timestamp = obj['LastModified'].replace(tzinfo=None)  # Remove timezone for comparison
                if key not in processed_objects or timestamp > processed_objects[key]:
                    if key not in processed_objects:
                        print(f"New object created: {key} - This one was added now")
                        if key.startswith('dog_'):
                            print("dog added")
          
                            response = s3_client.list_objects(Bucket=bucket_name)

                            images = [obj['Key'] for obj in response.get('Contents', []) if obj['Key'].lower().endswith(('.jpg', '.jpeg', '.png'))]

                            weights = models.resnet50(pretrained=True)
                            model = models.resnet50(weights=weights)
                            model.fc = nn.Identity()
                            model.eval()
                            transform = transforms.Compose([
                                transforms.Resize((224, 224)),
                                transforms.ToTensor()
                            ])
                            annoy_index = AnnoyIndex(2048, 'angular')
                            for i, image_name in enumerate(images):
                                # Download the image from the source bucket
                                file_name = f'./downloaded_images/{image_name}'  
                                s3_client.download_file(bucket_name, image_name, file_name)
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
                            s3_client_target = boto3.client('s3', aws_access_key_id='404_key_not_found', aws_secret_access_key='404_key_not_found', region_name='us-east-1')

                            # Upload the Annoy index file to the target bucket
                            with open(index_file_name, 'rb') as file:
                                s3_client_target.upload_fileobj(file, annoy_bucket, index_file_name)

                            # Clean up temporary files
                            os.remove(file_name)
                            os.remove(index_file_name)

                            print(f'Annoy index saved and uploaded to bucket: {annoy_bucket}')

                    processed_objects[key] = timestamp

if __name__ == "__main__":
    fetch_existing_objects()
    process_s3_events()
