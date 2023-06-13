import os
import cv2
import numpy as np

import torch
import segmentation_models_pytorch as smp

class ProcessorBinarySegmentation:

    OUT_DIR_MASKS_PROB = "masks_prob"
    OUT_DIR_MASKS_BINARY = "masks_bin"
    OUT_DIR_CONCAT = "vis_hconcat"
    OUT_DIR_OVERLAY = "vis_overlay"

    DEFAULT_KEY_MASK = "mask"
    DEFAULT_KEY_IMAGE = "image"
    def __init__(self, methods: list, path_output: str, confidence: float = 0.5):

        self.methods = methods
        self.path_output = path_output
        self.confidence = confidence

    def apply(self, batch_input, batch_output, batch_info):
        for m in self.methods:
            method = getattr(self, m)
            method(batch_input, batch_output, batch_info, self.path_output, self.confidence)

    @staticmethod
    def generate_masks_probability(batch_input, batch_output, batch_info, path_output, confidence):
        path_f = os.path.join(path_output, ProcessorBinarySegmentation.OUT_DIR_MASKS_PROB)
        if not os.path.exists(path_f): os.makedirs(path_f)

        batch_size, x, h, w = batch_output.shape
        for sample_id in range(0, batch_size):
            input_image = batch_input[0][sample_id]
            input_mask = batch_input[1][sample_id][0]
            predicted_mask = batch_output[sample_id][0]
            info = batch_info[sample_id]

            if ProcessorBinarySegmentation.DEFAULT_KEY_MASK not in info: continue

            output_mask = predicted_mask * 255

            file_name = os.path.basename(info[ProcessorBinarySegmentation.DEFAULT_KEY_MASK])
            file_path = os.path.join(path_f, file_name)

            cv2.imwrite(file_path, output_mask)

    @staticmethod
    def generate_masks_binary(batch_input, batch_output, batch_info, path_output, confidence):
        path_f = os.path.join(path_output, ProcessorBinarySegmentation.OUT_DIR_MASKS_BINARY)
        if not os.path.exists(path_f): os.makedirs(path_f)

        batch_size, x, h, w = batch_output.shape
        for sample_id in range(0, batch_size):
            input_image = batch_input[0][sample_id]
            input_mask = batch_input[1][sample_id][0]
            predicted_mask = batch_output[sample_id][0]
            info = batch_info[sample_id]

            if ProcessorBinarySegmentation.DEFAULT_KEY_MASK not in info: continue

            output_mask = np.zeros_like(predicted_mask)
            output_mask[predicted_mask < confidence] = 0
            output_mask[predicted_mask >= confidence] = 255

            file_name = os.path.basename(info[ProcessorBinarySegmentation.DEFAULT_KEY_MASK])
            file_path = os.path.join(path_f, file_name)

            cv2.imwrite(file_path, output_mask)

    @staticmethod
    def generate_hconcat(batch_input, batch_output, batch_info, path_output, confidence):
        path_f = os.path.join(path_output, ProcessorBinarySegmentation.OUT_DIR_CONCAT)
        if not os.path.exists(path_f): os.makedirs(path_f)

        batch_size, x, h, w = batch_output.shape
        for sample_id in range(0, batch_size):
            input_image = batch_input[0][sample_id]
            input_image = np.moveaxis(input_image, 0, -1)
            input_mask = batch_input[1][sample_id][0]
            predicted_mask = batch_output[sample_id][0]
            info = batch_info[sample_id]

            g_input_image = input_image * 255
            g_input_mask = input_mask * 255
            g_output_mask = predicted_mask * 255

            g_input_mask = cv2.cvtColor(g_input_mask, cv2.COLOR_GRAY2BGR)
            g_output_mask = cv2.cvtColor(g_output_mask, cv2.COLOR_GRAY2BGR)
            output = cv2.hconcat([g_input_image, g_input_mask, g_output_mask])

            file_name = os.path.basename(info[ProcessorBinarySegmentation.DEFAULT_KEY_MASK])
            file_path = os.path.join(path_f, file_name)
            cv2.imwrite(file_path, output)

    @staticmethod
    def compute_smp_metrics(batch_input, batch_output, batch_info, path_output, confidence):
        torch_masks_gt = torch.from_numpy(batch_input[1])
        torch_masks_pr = torch.from_numpy
