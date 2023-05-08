import os
import boto3


class ManagerAwsBucket:

    def __init__(self, aws_access_key: str = None, aws_secret_access_key: str = None, bucket_name: str = None):
        if aws_access_key is None or aws_secret_access_key is None:
            self.client = boto3.client("s3")
        else:
            self.client = boto3.client("s3", aws_access_key_id=aws_access_key, aws_secret_access_key=aws_secret_access_key)

        self.bucket_name = bucket_name

    def switch_bucket(self, bucket_name: str):
        self.bucket_name = bucket_name

    def upload_file(self, path_file_local: str, path_dir_remote: str = None, new_file_name: str = None) -> bool:
        """
        Upload file to given bucket to the specific remote directory, change file name if new name is provided
        :param path_file_local: absolute path to the file on the local machine
        :param path_dir_remote: directory path in the bucket
        :param new_file_name: new name of the file in the bucket
        :return: True if all ok, otherwise False
        """
        if self.bucket_name is None: return False

        try:
            if path_dir_remote is None: path_dir_remote = ""
            if new_file_name is None: new_file_name = os.path.basename(path_file_local)

            new_file_path = os.path.join(path_dir_remote, new_file_name)
            self.client.upload_file(path_file_local, self.bucket_name, new_file_path)
            return True
        except:
            return False