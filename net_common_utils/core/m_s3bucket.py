import os
import boto3

# --------------------------------------------------------------
# ---- AWS S3 Bucket access requires credentials provided by IAM
# ---- (aws_access_key_id and aws_secret_access_key)
# ---- For automatic verification of credentials create file
# ---- `/home/$user/.aws/credentials` with the following format
# [default]
# aws_access_key_id=<access_key>
# aws_secret_access_key=<secret_access_key>
# -------------------------------------------------------------
class ManagerAwsBucket:


    def __init__(self, aws_access_key: str = None, aws_secret_access_key: str = None, bucket_name: str = None, root: str = None):
        if aws_access_key is None or aws_secret_access_key is None:
            self.client = boto3.client("s3")
        else:
            self.client = boto3.client("s3", aws_access_key_id=aws_access_key, aws_secret_access_key=aws_secret_access_key)

        self.bucket_name = bucket_name
        self.root = root

    def _check_is_connection_established(func):
        """
        Decorator to check bucket connection
        """
        def _check(self, *args, **kwargs):
            if self.bucket_name is None: return False
            return func(self, *args, **kwargs)
        return _check

    def switch_bucket(self, bucket_name: str):
        """
        Switch to the new bucket
        :param bucket_name: the name of the bucket to switch to
        :return:
        """
        self.bucket_name = bucket_name

    @_check_is_connection_established
    def upload_file(self, path_file_local: str, path_dir_remote: str = None, new_file_name: str = None) -> bool:
        """
        Upload file to given bucket to the specific remote directory, change file name if new name is provided
        :param path_file_local: absolute path to the file on the local machine
        :param path_dir_remote: directory path in the bucket
        :param new_file_name: new name of the file in the bucket
        :return: True if all ok, otherwise False
        """

        try:
            if path_dir_remote is None: path_dir_remote = ""
            if new_file_name is None: new_file_name = os.path.basename(path_file_local)

            # ---- Path to the file in the bucket
            if self.root is None:
                new_file_path = os.path.join(path_dir_remote, new_file_name)
            else:
                new_file_path = os.path.join(self.root, path_dir_remote, new_file_name)


            self.client.upload_file(path_file_local, self.bucket_name, new_file_path)
            return True
        except:
            return False

    @_check_is_connection_established
    def download_file(self, path_file_remote: str, path_dir_local: str = None, new_file_name: str = None) -> bool:
        """
        Download file from s3 bucket to the given local directory and give it a new name
        :param path_file_remote: path to file in s3 bucket
        :param path_dir_local: path to local directory where the file should be saved
        :param new_file_name: new name of the downloaded file
        :return: True if all ok, otherwise False
        """

        try:
            if path_dir_local is None: path_dir_local = ""
            if new_file_name is None: new_file_name = os.path.basename(path_file_remote)

            new_file_path = os.path.join(path_dir_local, new_file_name)
            if self.root is not None:
                path_file_remote = os.path.join(self.root, path_file_remote)
            self.client.download_file(self.bucket_name, path_file_remote, new_file_path)
            return  True
        except Exception as e:
            print(e)
            return False

    @_check_is_connection_established
    def upload_dir(self, path_dir_local: str, path_dir_remote: str = None):
        return False

    @_check_is_connection_established
    def download_dir(self, path_dir_remote: str, path_dir_local: str = None):
        print(path_dir_remote)
        found_objects = self.client.list_objects_v2(Bucket=self.bucket_name, Prefix=path_dir_remote)
        if "Contents" not in found_objects: return False
        print(found_objects)
        found_objects = found_objects["Contents"]
        if len(found_objects) == 0: return False

        if not os.path.exists(path_dir_local):
            os.makedirs(path_dir_local)

        for obj in found_objects:
            remote_path = obj["Key"]

            local_path = remote_path.replace(path_dir_remote, path_dir_local + "/")

            if not os.path.exists(os.path.dirname(local_path)):
                os.makedirs(os.path.dirname(local_path))

            local_dir = os.path.dirname(local_path)
            is_ok = self.download_file(remote_path, local_dir)

        return True
