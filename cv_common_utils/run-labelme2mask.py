import os
import cv2
import json
import argparse

from core.cv2d_bmask import cv2d_convert_polygons2mask

# -----------------------------------------------------------------------------------------
class LabelmeFile:

    KEY_IMAGE_WIDTH = "imageWidth"
    KEY_IMAGE_HEIGHT = "imageHeight"
    KEY_OBJECTS = "shapes"
    KEY_LABEL = "label"
    KEY_POINTS = "points"
    KEY_IMAGE_PATH = "imagePath"

    def __init__(self):
        self.width = None
        self.height = None
        self.objects = []
        self.labels = []
    def load_from_file(self, path_labelme_json: str) -> bool:
        f = open(path_labelme_json, "r")
        data = json.load(f)
        f.close()

        try:

            self.width = data[self.KEY_IMAGE_WIDTH]
            self.height = data[self.KEY_IMAGE_HEIGHT]

            for obj in data[self.KEY_OBJECTS]:

                label = obj[self.KEY_LABEL]

                new_object = {"label": label, "points": []}
                if label not in self.labels:
                    self.labels.append(label)
                for point in obj[self.KEY_POINTS]:
                    x = point[0]
                    y = point[1]
                    new_object["points"].append([x,y])

                self.objects.append(new_object)
            return True
        except:
            return False

    def get_polygons(self, label_name: str = None) -> list:
        output = []
        for obj in self.objects:
            if label_name is not None and obj["label"] == label_name:
                output.append(obj["points"])
            if label_name is None:
                output.append(obj["points"])

        return output

class LabelmeDataset:

    def __init__(self):
        self.table = {}
        self.label_list = []

    def load_from_dir(self, path_dir: str):
        file_list = [f for f in os.listdir(path_dir) if os.path.isfile(os.path.join(path_dir, f)) and f.endswith(".json")]
        for file in file_list:
            file_path = os.path.join(path_dir, file)
            lf = LabelmeFile()
            is_ok = lf.load_from_file(file_path)

            if is_ok:
                file_key = file.replace(".json", "")
                self.table[file_key] = lf

                for label in lf.labels:
                    if label not in self.label_list:
                        self.label_list.append(label)

    @property
    def size(self):
        return len(self.table)

# -----------------------------------------------------------------------------------------

def generate_masks(path_dir_labelme: str, path_dir_mask: str):
    if not os.path.exists(path_dir_mask):
        os.makedirs(path_dir_mask)

    ld = LabelmeDataset()
    ld.load_from_dir(path_dir_labelme)

    for lf_key in ld.table:
        print("[INFO]: Generating masks from ", lf_key)
        lf = ld.table[lf_key]
        polygon_list = lf.get_polygons()

        mask = cv2d_convert_polygons2mask(lf.width, lf.height, polygon_list)
        path_mask = os.path.join(path_dir_mask, lf_key + ".png")
        cv2.imwrite(path_mask, mask)

# -----------------------------------------------------------------------------------------

INFO = (
    "Convert labelme JSON annotations into masks\n"
    "--in <path> - path to directory that contains JSON files produced by labelme\n"
    "--out <path> - path to directory where generated masks will be stored\n"
)

def parse_arguments():
    parser = argparse.ArgumentParser(description=INFO, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("-i", type=str, help="path to directory that contains JSON files produced by labelme")
    parser.add_argument("-o", type=str, help="path to directory that will contain generated masks")
    args = parser.parse_args()
    return args.i, args.o

# -----------------------------------------------------------------------------------------

if __name__ == '__main__':
    path_in, path_out = parse_arguments()
    generate_masks(path_in, path_out)
