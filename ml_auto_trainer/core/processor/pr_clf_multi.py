import json
import os
import copy
import cv2
import numpy as np

import torch
import segmentation_models_pytorch as smp

class ProcessorMultilabelClassification:

    OUT_FILE_PRED_LOGITS = "out-pred-prob.json"
    OUT_FILE_PRED_BINARY = "out-pred-binary.json"
    OUT_FILE_SCORES = "out-scores.json"

    DEFAULT_KEY_MASK = "mask"
    DEFAULT_KEY_IMAGE = "labels"

    def __init__(self, methods: list, path_output: str, confidence: float = 0.5):

        self.methods = methods
        self.path_output = path_output
        self.confidence = confidence

    def apply(self, batch_input, batch_output, batch_info):
        for m in self.methods:
            method = getattr(self, m)
            method(batch_input, batch_output, batch_info, self.path_output, self.confidence)


    @staticmethod
    def generate_predictions_probability(batch_input, batch_output, batch_info, path_output, confidence):
        if not os.path.exists(path_output): os.makedirs(path_output)

        # ---- File generation
        path_f = os.path.join(path_output, ProcessorMultilabelClassification.OUT_FILE_PRED_LOGITS)
        if not os.path.exists(path_f):
            f = open(path_f, "w")
            json.dump({"data": []}, f)
            f.close()

        f = open(path_f, "r")
        data_global = json.load(f)
        f.close()

        # ---- Write predictions
        batch_size = batch_output.shape[0]
        for sample_id in range(0, batch_size):
            input_image = batch_input[0][sample_id]
            labels_gt = copy.deepcopy(batch_input[1][sample_id])
            labels_pr = copy.deepcopy(batch_output[sample_id])
            info = copy.deepcopy(batch_info[sample_id])

            info["labels_gt"] = labels_gt.tolist()
            info["labels_pr"] = labels_pr.tolist()

            data_global["data"].append(info)

        f = open(path_f, "w")
        json.dump(data_global, f)
        f.close()

    @staticmethod
    def generate_predictions_binary(batch_input, batch_output, batch_info, path_output, confidence):
        if not os.path.exists(path_output): os.makedirs(path_output)

        # ---- File generation
        path_f = os.path.join(path_output, ProcessorMultilabelClassification.OUT_FILE_PRED_BINARY)
        if not os.path.exists(path_f):
            f = open(path_f, "w")
            json.dump({"data": []}, f)
            f.close()

        f = open(path_f, "r")
        data_global = json.load(f)
        f.close()

        # ---- Write predictions
        batch_size = batch_output.shape[0]
        for sample_id in range(0, batch_size):
            input_image = batch_input[0][sample_id]
            labels_gt = copy.deepcopy(batch_input[1][sample_id])
            labels_pr = copy.deepcopy(batch_output[sample_id])
            info = copy.deepcopy(batch_info[sample_id])

            labels_pr[labels_pr < confidence] = 0
            labels_pr[labels_pr >= confidence] = 1
            info["labels_gt"] = labels_gt.tolist()
            info["labels_pr"] = labels_pr.tolist()

            data_global["data"].append(info)

        f = open(path_f, "w")
        json.dump(data_global, f)
        f.close()


    @staticmethod
    def compute_scores(batch_input, batch_output, batch_info, path_output, confidence):
        if not os.path.exists(path_output): os.makedirs(path_output)

        # ---- Load from file
        path_f = os.path.join(path_output, ProcessorMultilabelClassification.OUT_FILE_SCORES)
        if not os.path.exists(path_f):
            f = open(path_f, "w")
            data = {}
            f.close()
        else:
            f = open(path_f, "r")
            data = json.load(f)
            f.close()


        batch_size = batch_output.shape[0]
        for sample_id in range(0, batch_size):
            input_image = batch_input[0][sample_id]
            labels_gt = copy.deepcopy(batch_input[1][sample_id])
            labels_pr = copy.deepcopy(batch_output[sample_id])
            info = copy.deepcopy(batch_info[sample_id])

            labels_pr[labels_pr < confidence] = 0
            labels_pr[labels_pr >= confidence] = 1

            for label_id in range(0, len(labels_pr)):
                if str(label_id) not in data: data[str(label_id)] = {"tp": 0, "tn": 0, "fp": 0, "fn": 0, "acc": 0, "f-score": 0}

            if len(labels_pr) > 1:
                label_id_gt = np.argmax(labels_gt)
                label_id_pr = np.argmax(labels_pr)

                for label_id in range(0, len(labels_pr)):
                    if label_id == label_id_gt and label_id == label_id_pr:
                        data[str(label_id)]["tp"] += 1
                    elif label_id == label_id_gt and label_id != label_id_pr:
                        data[str(label_id)]["fn"] += 1
                    elif label_id != label_id_gt and label_id == label_id_pr:
                        data[str(label_id)]["fp"] += 1
                    elif label_id != label_id_gt and label_id != label_id_pr:
                        data[str(label_id)]["tn"] += 1
            else:
                if labels_gt[0] == 1 and labels_pr[0] == 1:
                    data["0"]["tp"] += 1
                elif labels_gt[0] == 0 and labels_pr[0] == 1:
                    data["0"]["fp"] += 1
                if labels_gt[0] == 1 and labels_pr[0] == 0:
                    data["0"]["fn"] += 1
                if labels_gt[0] == 0 and labels_pr[0] == 0:
                    data["0"]["tn"] += 1


            if "global" not in data:
                data["global"] = {"samples_correct": 0, "samples_count": 0}
            if label_id_gt == label_id_pr:
                data["global"]["samples_correct"] += 1
            data["global"]["samples_count"] += 1

        f = open(path_f, "w")
        json.dump(data, f, indent=4)
        f.close()
