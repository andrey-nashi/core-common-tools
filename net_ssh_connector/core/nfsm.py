import os
import yaml
import getpass


class NetworkFileSystemManager:

    NETWORK_MAP_FILE = "settings.yaml"

    KEY_SSH_IDENTITY_FILE = "SSH_IDENTITY_FILE"

    KEY_NETWORK_MAP = "NETWORK_MAP"
    KEY_REMOTE_IP = "IP"
    KEY_REMOTE_USER = "USER"
    KEY_REMOTE_PORT = "PORT"

    KEY_MOUNT_POINTS = "MOUNT_POINTS"

    def __init__(self):
        self.path_resources = os.path.abspath(os.path.dirname( __file__ ))
        self.path_resources = os.path.join(self.path_resources, "resources")

        f = open(os.path.join(self.path_resources, self.NETWORK_MAP_FILE))
        data = yaml.load(f)
        f.close()

        # ---- Set the identity file if specified
        self.ssh_identity_file = data[self.KEY_SSH_IDENTITY_FILE]
        if self.ssh_identity_file is not None:
            self.ssh_identity_file_path = os.path.join(self.path_resources, self.ssh_key)
        else:
            self.ssh_identity_file_path = None

        self.network_map = data[self.KEY_NETWORK_MAP]
        self.mount_points = data[self.KEY_MOUNT_POINTS]

    def execute_mount(self, mount_point_name: str) -> bool:
        if mount_point_name not in self.mount_points:
            print("[ERROR]: No location", mount_point_name)
            return False

        mount_settings = self.mount_points[mount_point_name]
        machine_name = mount_settings[0]
        path_local = mount_settings[1]
        path_remote = mount_settings[2]

        if machine_name not in self.network_map:
            print("[ERROR]: No machine with name", machine_name)
            return False

        connection_settings = self.network_map[machine_name]
        remote_ip = connection_settings[self.KEY_REMOTE_IP]
        remote_user = connection_settings[self.KEY_REMOTE_USER]
        remote_port = connection_settings[self.KEY_REMOTE_PORT]

        if remote_user is None:
            remote_user = getpass.getuser()

        command = "sshfs "
        if self.ssh_identity_file_path is not None:
            command += " -o IdentityFile=" + self.ssh_identity_file_path
        if remote_port is not None:
            command += "-p " + remote_port + " "

        command += remote_user + "@" + remote_ip + ":" + path_remote + " " + path_local

        if not os.path.exists(path_local):
            os.makedirs(path_local)

        #command += "-o reconnect,transform_symlinks,allow_other"

        os.system(command)

    def execute_ssh(self, machine_name: str) -> bool:
        if machine_name not in self.network_map:
            print("[ERROR]: No machine with name", machine_name)
            return False

        try:
            connection_settings = self.network_map[machine_name]
            remote_ip = connection_settings[self.KEY_REMOTE_IP]
            remote_user = connection_settings[self.KEY_REMOTE_USER]
            remote_port = connection_settings[self.KEY_REMOTE_PORT]

            # ---- Fabricate ssh command
            command = "ssh "
            if self.ssh_identity_file_path is not None:
                command += "-i " + self.ssh_identity_file_path + " "
            if remote_port is not None:
                command += "-p " + remote_port + " "
            if remote_user is None:
                remote_user = getpass.getuser()
            command += remote_user + "@" + remote_ip
            os.system(command)
            return True
        except:
            print("[ERROR]: Exception occurred during ssh connection to", machine_name)
            return False

    def get_info_machines(self):
        for machine in self.network_map:
            print(machine, self.network_map[machine])

    def get_info_mountpoints(self):
        for mountpoint in self.sshfs_map:
            print(mountpoint, self.sshfs_map[mountpoint])
