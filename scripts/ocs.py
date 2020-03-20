import boto3
import json
import os
import urllib3
from kafka import KafkaProducer

def main():
    urllib3.disable_warnings()
    s3_access_key = os.environ['ACCESS_KEY_ID']
    s3_secret_key = os.environ['SECRET_ACCESS_KEY']
    s3_bucket_name = os.environ['S3_BUCKET']
    s3_endpoint_url = os.environ['S3_ENDPOINT_URL']
    model_filename = os.environ['MODEL_FILENAME']
    kafka_bootstrap = os.environ['KAFKA_BOOTSTRAP']
    kafka_topic = os.environ['KAFKA_TOPIC']
    s3 = boto3.client('s3',endpoint_url= s3_endpoint_url,
                           aws_access_key_id = s3_access_key,
                           aws_secret_access_key = s3_secret_key,
                           verify=False)
    msg = {}
    for key in s3.list_objects(Bucket=s3_bucket_name)['Contents']:
        print(key['Key'])
        if key['Key'] == model_filename:
            timestamp = key['LastModified'].timestamp()
            msg = {'modelName': model_filename, 'timestamp': timestamp, 's3Bucket': s3_bucket_name}
            break
    producer = KafkaProducer(bootstrap_servers=kafka_bootstrap)
    producer.send(kafka_topic, json.dumps(msg).encode('utf-8'))
    producer.flush()

# For more documentation on Producer: https://pypi.org/project/kafka-python/
if __name__ == '__main__':
    main()
