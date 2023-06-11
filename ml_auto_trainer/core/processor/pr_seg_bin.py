import os
import cv2

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
        batch_size, x, h, w = batch_output.shape
        for sample_id in range(0, batch_size):
            input_image = batch_input[0][sample_id]
            input_mask = batch_input[1][sample_id]
            output_mask = batch_output[sample_id]
            info = batch_info[sample_id]

            if len(output_mask.shape) == 3:
                output_mask = output_mask[0]
            output_mask = output_mask * 255

            if not os.path.exists(path_output): os.makedirs(path_output)
            file_name = os.path.basename(info["mask"])
            file_path = os.path.join(path_output, file_name)

            cv2.imwrite(file_path, output_mask)
