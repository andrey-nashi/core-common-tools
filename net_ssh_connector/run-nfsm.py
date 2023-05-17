#!/usr/bin/python3
from core import NetworkFileSystemManager
import argparse

# -------------------------------------------------------------------------------

INFO = (
    "Mount directory or ssh to a specific machine. Accepted commands\n"
    "* ssh <machine_name> <user> - ssh to a specific machine\n"
    "* mount <named_connection> <user> - mount a drive via named connection\n"
    "* info-mount - get info about named connections\n"
    "* info-ssh - get info about machines\n"
)

# -------------------------------------------------------------------------------


# -------------------------------------------------------------------------------

def run(args: list):
    if args is None or len(args) == 0:
        print(INFO)
        return

    nfsm = NetworkFileSystemManager()
    command = args[0]

    if command == "ssh" and len(args) >= 2:
        machine_name = args[1]
        nfsm.execute_ssh(machine_name)

    elif command == "mount" and len(args) >= 2:
        mountpoint = args[1]
        nfsm.execute_mount(mountpoint)

    elif command == "info-mount":
        nfsm.get_info_mountpoints()

    elif command == "info-ssh":
        nfsm.get_info_machines()


def parse_arguments():
    parser = argparse.ArgumentParser(description=INFO, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("command", type=str, nargs="*", help="command and its arguments")
    args = parser.parse_args()

    return args.command


if __name__ == "__main__":
    c = parse_arguments()
    run(c)


