import boto3
import json
import os
import sys
import urllib3
from kafka import KafkaProducer

def get_s3_object(access_key, secret_key, bucket, url, object_name):
    s3 = boto3.client('s3',endpoint_url=url,
                           aws_access_key_id=access_key,
                           aws_secret_access_key=secret_key,
                           verify=False)
    print("Bucket contents:")
    for key in s3.list_objects(Bucket=bucket)['Contents']:
        print(key['Key'])
        if key['Key'] == object_name:
            return key
            break
    return None

def create_json_message(object, s3_bucket_name):
    timestamp = object['LastModified'].timestamp()
    msg = {'modelName': object['Key'], 'timestamp': timestamp, 's3Bucket': s3_bucket_name}
    return msg

def send_kafka_message(producer, topic, payload):
    producer.send(topic, payload)
    producer.flush()

def main():
    urllib3.disable_warnings()

    s3_access_key = os.environ['ACCESS_KEY_ID']
    s3_secret_key = os.environ['SECRET_ACCESS_KEY']
    s3_bucket_name = os.environ['S3_BUCKET']
    s3_endpoint_url = os.environ['S3_ENDPOINT_URL']
    model_filename = os.environ['MODEL_FILENAME']
    kafka_bootstrap = os.environ['KAFKA_BOOTSTRAP']
    kafka_topic = os.environ['KAFKA_TOPIC']

    model_object = get_s3_object(s3_access_key, s3_secret_key, s3_bucket_name,
                                 s3_endpoint_url, model_filename)

    if model_object == None:
        print('Error: S3 object name ' + model_filename + ' not found')
        sys.exit(1)

    message = create_json_message(model_object, s3_bucket_name)
    print('Sending message:', message)

    producer = KafkaProducer(bootstrap_servers=kafka_bootstrap, acks='all')
    send_kafka_message(producer, kafka_topic, json.dumps(message).encode('utf-8'))
    print('Message sent')

# For more documentation on Producer: https://pypi.org/project/kafka-python/
if __name__ == '__main__':
    main()
