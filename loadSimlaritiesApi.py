
from flask import Flask, request, jsonify
import torch
from PIL import Image
from torchvision import models, transforms
import torch.nn as nn
from annoy import AnnoyIndex
import boto3
import io
app = Flask(__name__)

aws_access_key_id = '404_key_not_found'
aws_secret_access_key = '404_key_not_found'
images_bucket = 'petspulseimages'
annoy_bucket = 'similarityfeatureanns'


# Create an S3 client
s3_client = boto3.client('s3', aws_access_key_id=aws_access_key_id, aws_secret_access_key=aws_secret_access_key, region_name='us-east-1')

# Load the Annoy index from the Annoy bucket
def load_annoy_index_from_bucket():
    annoy_index = AnnoyIndex(2048, 'angular')
    s3_client.download_file(annoy_bucket, 'dog_index_moreacc.ann', '/tmp/dog_index_moreacc.ann')
    annoy_index.load('/tmp/dog_index_moreacc.ann')
    return annoy_index

# Load the ResNet model
weights = models.resnet50(pretrained=True)
model = models.resnet50(weights=weights)
model.fc = nn.Identity()
model.eval()

# Define the image transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])
response = s3_client.list_objects(Bucket=images_bucket)

# Create a mapping between image names and order indices
image_order_mapping = {}
order_index = 0

if 'Contents' in response:
    for obj in response['Contents']:
        image_key = obj['Key']
        image_order_mapping[image_key] = order_index
        order_index += 1


# API endpoint to find similar images
@app.route('/find_similar_images', methods=['POST'])
def find_similar_images():
    # Load the input image from the request
    input_image_file = request.files['image']
    input_image = Image.open(io.BytesIO(input_image_file.read()))
    input_tensor = transform(input_image).unsqueeze(0)

    # Load the Annoy index
    annoy_index = load_annoy_index_from_bucket()

    # Calculate the nearest neighbors of the input image
    output_tensor = model(input_tensor)
    nns, distances = annoy_index.get_nns_by_vector(output_tensor[0], 2, include_distances=True)
    # Return the names of the two most similar images
    similar_images_info = []
    for nn_index in nns:
        for image_name, index in image_order_mapping.items():
            if index == nn_index:
                similar_images_info.append({'image_name': image_name, 'index': index})

    # Return the names and indices of the two most similar images
    return jsonify({'similar_images_info': similar_images_info})

if __name__ == '__main__':
    app.run(port=8000)
