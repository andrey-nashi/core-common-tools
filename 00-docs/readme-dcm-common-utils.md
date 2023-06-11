**dcm_common_utils**

Utility methods for processing DICOM images

| Method name                           | Package            | Description                                           |
|---------------------------------------|--------------------|-------------------------------------------------------|
| DicomConverter.dcm2png_8bit_canonical | core.dcm_converter | Convert DICOM to 8-bit image, apply rescale intercept |
| DicomConverter.dcm2png_16bit_basic    | core.dcm_converter | Convert DICOM to 16-bit image                         |
| DicomConverter.dcm2png_8bit_equalized | core.dcm_converter | Convert DICOM to 8-bit image, apply equalization      |
| DicomConverter.dcm2png_8bit_16bitnorm | core.dcm_converter | Convert DICOM to 16-bit, then scale to 8-bit          |
| DicomConverter.dcm2png_8bit_clipped   | core.dcm_converter | Convert DICOM to 8-bit with clipping                  |
| DicomConverter.png2dcm_grayscale      | core.dcm_converter | Convert PNG to DICOM with some default tags           |
| DicomConverter.dcm_alter_tag_data     | core.dcm_converter | Change tag of the DICOM, output DICOM                 |