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


