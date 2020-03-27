import boto3
import json
import os
import requests
import sys
import tempfile
import urllib3

sas_omm_protocol = 'http'

def s3_client(access_key, secret_key, url):
    if s3_client.s3_client != None:
        return s3_client.s3_client

    s3_client.s3_client = boto3.client('s3', endpoint_url=url,
                                       aws_access_key_id=access_key,
                                       aws_secret_access_key=secret_key,
                                       verify=False)
    return s3_client.s3_client

s3_client.s3_client = None

def get_s3_object(access_key, secret_key, bucket, url, object_name):
    s3 = s3_client(access_key, secret_key, url)
    print("Bucket contents:")
    for key in s3.list_objects(Bucket=bucket)['Contents']:
        print(key['Key'])
        if key['Key'] == object_name:
            return key
            break
    return None

def get_s3_object_file(access_key, secret_key, bucket, url, object_name):
    s3 = s3_client(access_key, secret_key, url)
    fp = tempfile.TemporaryFile()
    s3.download_fileobj(Bucket=bucket, Key=object_name, Fileobj=fp)
    fp.seek(0)
    return fp

def sas_omm_login(server, user, password):
    # Login to REST services
    auth_uri = '/SASLogon/oauth/token'

    headers = {
       'Accept': 'application/json',
       'Content-Type': 'application/x-www-form-urlencoded'
    }

    payload = 'grant_type=password&username=' + user + '&password=' + password
    auth_return = requests.post(sas_omm_protocol + '://' + server + auth_uri ,
                                auth=('sas.ec', ''), data=payload, headers=headers);
    print("OMM Login:", auth_return)
    auth_json = json.loads(auth_return.content.decode('utf-8'))
    return auth_json['access_token']

def sas_omm_get_model_repository(server, auth_token):
    # Get Model Repository
    headers = {'Authorization': 'Bearer ' + auth_token}

    url = sas_omm_protocol + '://' + server + "/modelRepository/repositories?filter=eq(name,'Public')"
    repo_list = requests.get(url, headers=headers)
    repo = repo_list.json()['items'][0]
    repo_id = repo['id']
    repo_folder_id = repo['folderId']
    return repo_id, repo_folder_id

def sas_omm_find_project(server, auth_token, project_name):
    project_id = -1
    project = None
    headers={
      'content-type': 'application/vnd.sas.models.project+json',
      'Authorization': 'Bearer ' + auth_token
    }

    url = sas_omm_protocol + '://' + server + "/modelRepository/projects?filter=eq(name, '" + project_name + "')"
    project_result = requests.get(url, headers = headers)

    if project_result:
        print("OMM Find Project Result JSON Model", project_result.json())
        if project_result.json()['count'] > 0:
            project = project_result.json()['items'][0]
            project_id = project['id']
        else:
            project_result = False

    return project_result, project_id, project

def sas_omm_create_project(server, auth_token, project_name, repo_id, repo_folder_id):
    project_id = -1
    headers={
      'content-type': 'application/vnd.sas.models.project+json',
      'Authorization': 'Bearer ' + auth_token
    }
    new_project={
        'name': project_name,
        'repositoryId': repo_id,
        'folderId': repo_folder_id
    }

    url = sas_omm_protocol + '://' + server + '/modelRepository/projects'
    project_result = requests.post(url, data=json.dumps(new_project), headers=headers)

    if project_result:
        project = project_result.json()
        project_id = project['id']

    return project_result, project_id

def sas_omm_create_model(server, auth_token, project_id, model_name):
    # Create the model
    headers={
      'content-type': 'application/vnd.sas.models.model+json',
      'Authorization': 'Bearer ' + auth_token
    }
    new_model={
            'name': model_name,
            'projectId': project_id,
            'function': 'classification',
            'scoreCodeType': 'python'
    }

    url = sas_omm_protocol + '://' + server + '/modelRepository/models'
    model_result = requests.post(url, data=json.dumps(new_model), headers=headers)
    if model_result.status_code == requests.codes.created:
        model = model_result.json()
        model_id = model['items'][0]['id']
        return model_result, model_id, model

    return model_result, None, None

def sas_omm_import_model(server, auth_token, model_id, model_file):
    # Import files into the model
    headers={
      'Content-Type': 'application/octet-stream',
      'Authorization': 'Bearer ' + auth_token
    }
    url = sas_omm_protocol + '://' + server + '/modelRepository/models/' + model_id + '/contents?name=model_final.pth'
    model_file_result = requests.post(url, data=model_file, headers=headers)

def sas_omm_publish_model(server, user, password, project_name, model_name, model_file):
    auth_token = sas_omm_login(server, user, password)
    if auth_token == None:
        return False

    repo_id, repo_folder_id = sas_omm_get_model_repository(server, auth_token)
    if repo_id == None or repo_folder_id == None:
        return False

    print("OMM Repository ID: " + repo_id)
    print("OMM Repository Folder ID: " + repo_folder_id)
    print("OMM Project Name: " + project_name)
    project_result, project_id, project = sas_omm_find_project(server, auth_token, project_name)
    print("OMM Find Project Result:", project_result)

    if project_result == False:
        project_result, project_id = sas_omm_create_project(server, auth_token,
                                                            project_name,
                                                            repo_id,
                                                            repo_folder_id)
        print("OMM Create Project Result:", project_result)
        print("OMM Project ID", project_id)
    else: # Found existing project, so use it.
        # FIXME: Return False as the next create model fails with 401 conflict
        # if model already exists. Need to add update model logic.
        print("OMM Project ID", project_id)
        print("OMM Update Model Result:", project_result)
        return False


    model_result, model_id, model = sas_omm_create_model(server, auth_token, project_id, model_name)
    print("OMM Create Model Result:", model_result)
    if model_result.status_code != requests.codes.created:
        return False

    print("OMM JSON Model", model)
    print("OMM Model ID: " + model_id)

    model_file_result = sas_omm_import_model(server, auth_token, model_id, model_file)
    if model_file_result == None:
        return False

    print("OMM Model File Import Result ", model_file_result)

    return True


def main():
    urllib3.disable_warnings()

    s3_access_key = os.environ['ACCESS_KEY_ID']
    s3_secret_key = os.environ['SECRET_ACCESS_KEY']
    s3_bucket_name = os.environ['S3_BUCKET']
    s3_endpoint_url = os.environ['S3_ENDPOINT_URL']
    model_filename = os.environ['MODEL_FILENAME']
    sas_omm_server = os.environ['SAS_OMM_SERVER']
    sas_omm_user = os.environ['SAS_OMM_USER']
    sas_omm_password = os.environ['SAS_OMM_PASSWORD']
    sas_omm_project_name = os.environ['SAS_OMM_PROJECT_NAME']
    sas_omm_model_name = os.environ['SAS_OMM_MODEL_NAME']

    model_object = get_s3_object(s3_access_key, s3_secret_key, s3_bucket_name,
                                 s3_endpoint_url, model_filename)

    if model_object == None:
        print('Error: S3 object name ' + model_filename + ' not found')
        sys.exit(1)

    model_file = get_s3_object_file(s3_access_key, s3_secret_key, s3_bucket_name,
                                    s3_endpoint_url, model_filename)

    if model_file == None:
        print('Error: Unable to download S3 object file ' + model_filename)
        sys.exit(1)

    success = sas_omm_publish_model(sas_omm_server, sas_omm_user, sas_omm_password,
                                    sas_omm_project_name, sas_omm_model_name, model_file)
    if not success:
        print('Error: unable to publish AI model ' + model_filename + ' to SAS OMM')
        model_file.close()
        sys.exit(1)

    print('Successfully published AI model ' + model_filename + ' to SAS OMM')
    model_file.close()

if __name__ == '__main__':
    main()
