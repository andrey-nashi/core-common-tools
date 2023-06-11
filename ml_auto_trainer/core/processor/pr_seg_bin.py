import os
import cv2
import numpy as np
class ProcessorBinarySegmentation:

    def __init__(self, methods: list, path_output: str):

        self.methods = methods
        self.path_output = path_output

    def apply(self, batch_input, batch_output, batch_info):
        for m in self.methods:
            method = getattr(self, m)
            method(batch_input, batch_output, batch_info, self.path_output)


    @staticmethod
    def generate_masks(batch_input, batch_output, batch_info, path_output):
        path_f = os.path.join(path_output, "masks")
        if not os.path.exists(path_f): os.makedirs(path_f)

        batch_size, x, h, w = batch_output.shape
        for sample_id in range(0, batch_size):
            input_image = batch_input[0][sample_id]
            input_mask = batch_input[1][sample_id][0]
            output_mask = batch_output[sample_id][0]
            info = batch_info[sample_id]

            print(np.min(output_mask), np.max(output_mask))
            output_mask = output_mask * 255

            file_name = os.path.basename(info["mask"])
            file_path = os.path.join(path_f, file_name)

            cv2.imwrite(file_path, output_mask)

    @staticmethod
    def generate_hconcat(batch_input, batch_output, batch_info, path_output):
        path_f = os.path.join(path_output, "hconcat")
        if not os.path.exists(path_f): os.makedirs(path_f)

        batch_size, x, h, w = batch_output.shape
        for sample_id in range(0, batch_size):
            input_image = batch_input[0][sample_id]
            input_image = np.moveaxis(input_image, 0, -1)
            input_mask = batch_input[1][sample_id][0]
            output_mask = batch_output[sample_id][0]
            info = batch_info[sample_id]

            input_image = input_image * 255
            input_mask = input_mask * 255
            output_mask = output_mask * 255

            input_mask = cv2.cvtColor(input_mask, cv2.COLOR_GRAY2BGR)
            output_mask = cv2.cvtColor(output_mask, cv2.COLOR_GRAY2BGR)
            output = cv2.hconcat([input_image, input_mask, output_mask])

            file_name = os.path.basename(info["mask"])
            file_path = os.path.join(path_f, file_name)
            cv2.imwrite(file_path, output)



