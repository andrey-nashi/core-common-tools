import os
import argparse


class BasicCliApp:

    def __init__(self, app_info: str = "", app_args: dict = None):
        self.app_info = app_info
        self.app_args = app_args



    def _parse_arguments(self):
        parser = argparse.ArgumentParser(description="", formatter_class=argparse.RawTextHelpFormatter)
        parser.add_argument("-i", type=str, help="path to directory that contains JSON files produced by labelme")
        args = parser.parse_args()
        return args

    def run(self):
        return



class InteractiveCliApp:

    def __init__(self):
        return

    def run(self):
        return