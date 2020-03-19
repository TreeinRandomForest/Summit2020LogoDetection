import boto3
import os

s3_access_key = os.environ['ACCESS_KEY_ID']
s3_secret_key = os.environ['SECRET_ACCESS_KEY']
s3_bucket_name = os.environ['S3_BUCKET']
s3_endpoint_url = os.environ['S3_ENDPOINT_URL']

s3 = boto3.client('s3',endpoint_url= s3_endpoint_url,
                       aws_access_key_id = s3_access_key,
                       aws_secret_access_key = s3_secret_key,
                       verify=False)
for key in s3.list_objects(Bucket=s3_bucket_name)['Contents']:
    print(key['Key'])
