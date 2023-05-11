import numpy


def convert_file_bin2png(path_file: str):
    f = open(path_file, "rb")
    compressed_array = f.read()
    f.close()


    return blosc.unpack_array(compressed_array)
