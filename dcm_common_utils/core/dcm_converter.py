import os
import cv2
import numpy as np
from skimage import exposure

import pydicom
from pydicom.dataset import Dataset, FileDataset
from pydicom.uid import ExplicitVRLittleEndian
import pydicom._storage_sopclass_uids

#---------------------------------------------------------
class DicomConverter:

    # ---- DICOM 2 PNG conversion constants
    # ---- * DCM2PNG_MODE_8BIT_CANONICAL - convert to 8-bit image with canonical linear transformation of slope, intercept
    # ---- * DCM2PNG_MODE_8BIT_EQUALIZED - convert to 8-bit image with scaling from 16-bit and additional histogram equalization
    # ---- * DCM2PNG_MODE_8BIT_F16BIT - convert to 8-bit image with direct min/max scaling from 16-bit
    # ---- * DCM2PNG_MODE_8BIT_CLIPPED - convert to 8-bit image with rescale/intercept applied and additional clipping
    # ---- * DCM2PNG_MODE_16BIT - convert to 16-bit, no scaling, raw pixels

    DCM2PNG_MODE_8BIT_CANONICAL = 0
    DCM2PNG_MODE_8BIT_EQUALIZED = 1
    DCM2PNG_MODE_8BIT_F16BIT = 2
    DCM2PNG_MODE_8BIT_CLIPPED = 3
    DCM2PNG_MODE_16BIT = 4

    @staticmethod
    def dcm2png_8bit_canonical(path: str):
        """
        A standard method for converting DICOM to PNG files.
        The correctness of the method was evaluated using ImageJ
        """
        try:
            dcm = pydicom.dcmread(path, force=True)
            img = dcm.pixel_array

            isInverse = False

            try:
                rescaleSlope = dcm.RescaleSlope
                rescaleIntercept = dcm.RescaleIntercept
                wCenter = dcm.WindowCenter
                wWidth = dcm.WindowWidth
                wlow = wCenter - wWidth / 2
                whigh = wCenter + wWidth / 2

            except:
                rescaleIntercept = 0
                rescaleSlope = 1
                wlow = np.min(img)
                whigh = np.max(img)

            if dcm.PhotometricInterpretation == "MONOCHROME1":
                isInverse = True

            img = (img * rescaleSlope) + rescaleIntercept

            imgx = cv2.convertScaleAbs(img - wlow, alpha=(255.0 / (whigh - wlow)))
            imgx[img < 0] = 0
            if isInverse: imgx = 255 - imgx

            return imgx
        except:
            print("EXCEPTION: Can't process file:  " + path)
            return None

    @staticmethod
    def dcm2png_16bit_basic(path: str):
        """
        A standard method for converting DICOM to PNG files.
        The correctness of the method was evaluated using ImageJ
        """
        try:
            dcm = pydicom.dcmread(path, force=True)
            img = dcm.pixel_array.astype(np.uint16)

            if dcm.PhotometricInterpretation == "MONOCHROME1":
                img = np.max(img) - img

            return img
        except:
            print("EXCEPTION: Can't process file:  " + path)
            return None

    @staticmethod
    def dcm2png_8bit_equalized(path: str):
        try:
            dcm = pydicom.dcmread(path)
            image = dcm.pixel_array.astype(np.uint16)
            assert image.dtype == np.uint16

            if dcm.PhotometricInterpretation == "MONOCHROME1":
                image = np.max(image) - image

            image = image.astype(np.float64)
            image = exposure.equalize_hist(image, nbins=255)
            image *= 255
            image = image.astype(np.uint8)
            return image
        except:
            print("EXCEPTION: Can't process file:  " + path)
            return None

    @staticmethod
    def dcm2png_8bit_16bitnorm(path: str):
        try:
            dcm = pydicom.dcmread(path, force=True)
            img = dcm.pixel_array.astype(np.uint16)

            if dcm.PhotometricInterpretation == "MONOCHROME1":
                img = np.max(img) - img

            img = (img - np.min(img)) / (np.max(img) - np.min(img)) * 255

            return img
        except:
            print("EXCEPTION: Can't process file:  " + path)
            return None

    @staticmethod
    def dcm2png_8bit_clipped(path):
        """
        A standard method for converting DICOM to PNG files.
        The correctness of the method was evaluated using ImageJ
        """
        dcm = pydicom.dcmread(path, force=True)
        img = dcm.pixel_array

        isInverse = False

        try:
            rescaleSlope = dcm.RescaleSlope
            rescaleIntercept = dcm.RescaleIntercept
            wCenter = dcm.WindowCenter
            wWidth = dcm.WindowWidth
            wlow = wCenter - wWidth / 2
            whigh = wCenter + wWidth / 2

        except:
            rescaleIntercept = 0
            rescaleSlope = 1
            wlow = np.min(img)
            whigh = np.max(img)

        if dcm.PhotometricInterpretation == "MONOCHROME1":
            isInverse = True

        img = (img * rescaleSlope) + rescaleIntercept
        if (whigh - wlow) != 0: imgx = (img - wlow) / (whigh - wlow) * 255
        else: imgx = (img - wlow) / whigh * 255
        imgx = np.round(np.clip(imgx, 0, 255))
        if isInverse: imgx = 255 - imgx

        image = imgx.astype(np.uint8)
        return image

    @staticmethod
    def png2dcm_grayscale(path_input: str, path_output: str, modality_tag: str, uid_vendor: str):
        """
        :param path_input: path to the input PNG file
        :param path_output: path to the output DICOM file
        :param modality_tag: modality tag such as CR, DX, CT, MR to initialize in the output DICOM
        :param uid_vendor:
        """
        file_name = os.path.basename(path_input)
        patient_name = file_name.replace(".png", "")

        image = cv2.imread(path_input)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        meta = pydicom.dataset.FileMetaDataset()
        meta.FileMetaInformationGroupLength = 4
        meta.MediaStorageSOPClassUID = pydicom._storage_sopclass_uids.SecondaryCaptureImageStorage
        meta.MediaStorageSOPInstanceUID = pydicom.uid.generate_uid()
        meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian

        ds = Dataset()
        ds.file_meta = meta
        ds.is_little_endian = True
        ds.is_implicit_VR = False

        ds.SOPClassUID = meta.MediaStorageSOPClassUID
        ds.PatientName = patient_name
        ds.PatientID = patient_name

        ds.SeriesInstanceUID = pydicom.uid.generate_uid(prefix=uid_vendor)
        ds.StudyInstanceUID = pydicom.uid.generate_uid(prefix=uid_vendor)
        ds.FrameOfReferenceUID = pydicom.uid.generate_uid(prefix=uid_vendor)
        ds.SOPInstanceUID = pydicom.uid.generate_uid(prefix=uid_vendor)

        ds.BitsStored = 8
        ds.BitsAllocated = 8
        ds.SamplesPerPixel = 1
        ds.HighBit = 7
        ds.Rows = image.shape[0]
        ds.Columns = image.shape[1]
        ds.InstanceNumber = 1

        ds.RescaleIntercept = "0"
        ds.RescaleSlope = "1"
        ds.PixelSpacing = r"1\1"
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.PixelRepresentation = 0
        ds.Modality = modality_tag
        ds.PixelData = image.tobytes()

        pydicom.dataset.validate_file_meta(ds.file_meta, enforce_standard=True)
        pydicom.filewriter.dcmwrite(path_output, ds, write_like_original=False)

    @staticmethod
    def dcm_alter_tag_data(path_in: str, path_out: str, tag_data: dict):
        """
        Alter DICOM tags
        :param path_in: path to input DICOM file
        :param path_out: path to output DICOM file
        :param tag_data: dictionary of paris tag: value
        * if the tag==$filename - then the tag is assigned the base filename
        :return:
        """
        try:
            dcm = pydicom.dcmread(path_in, force=True)

            for key in tag_data:
                value = tag_data[key]
                if value == "$filename":
                    value = os.path.basename(path_in).replace(".dcm", "")
                if value == "$filename_base":
                    value = os.path.basename(path_in).replace(".dcm", "")
                    value = value.split("_")[0]

                if key in dcm:
                    tag = dcm.data_element(key).tag
                    dcm[tag].value = value

            pydicom.filewriter.dcmwrite(path_out, dcm)

        except:
            print("EXCEPTION: Can't process file:  " + path_in)

