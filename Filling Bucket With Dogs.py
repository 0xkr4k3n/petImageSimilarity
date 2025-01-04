import boto3
import os
import uuid

# AWS credentials
access_key = '404_key_not_found'
secret_key = '404_key_not_found'
region = 'us-east-1'


# Bucket name
bucket_name = 'petpulse'

# Local directory containing the images
local_directory = 'Dog'

# Create an S3 client
s3_client = boto3.client('s3', aws_access_key_id=access_key, aws_secret_access_key=secret_key, region_name=region)

# Upload images to the S3 bucket with unique names
image_mapping = {}  # Dictionary to store original and new image names

for i in range(500):
    original_image_name = f'{i}.jpg'  # Original image name
    new_image_name = f'dog_{uuid.uuid4().hex}.jpg'  # New image name with prefix
    
    # Upload the image to the S3 bucket
    s3_client.upload_file(os.path.join(local_directory, original_image_name), bucket_name, new_image_name)
    
    # Add the mapping to the dictionary
    image_mapping[original_image_name] = new_image_name
    
    print(f'Uploaded: {original_image_name} as {new_image_name}')

print('Upload complete.')

# Print the mapping of original and new image names
print('Image Mapping:')
for original_name, new_name in image_mapping.items():
    print(f'{original_name} -> {new_name}')

