**cv_common_utils**

| Function name                  | Package            | Description                                                               |
|--------------------------------|--------------------|---------------------------------------------------------------------------|
| cv2d_convert_bbox2mask         | core.cv2d_convert  | Generate mask paining given list of bounding boxes with intensity score   |
| cv2d_convert_mask2bbox         | core.cv2d_convert  | Calculate bounding boxes of connected components in a binary mask         |
| cv2d_convert_polygon2mask      | core.cv2d_convert  | Generate masks painting given polygon wih specified intensity             |
| cv2d_convert_polygons2mask     | core.cv2d_convert  | Generate masks painting given list polygons wih specified intensity       |
| cv2d_convert_mask2polygon      | core.cv2d_convert  | Calculate polygons for connected components of a mask                     |
| cv2d_convert_pix2label         | core.cv2d_convert  | Convert RGB image to label map using the given LUT                        |
| cv2d_convert_label2pix         | core.cv2d_convert  | Convert label map to RGB image using the given LUT                        |
| cv2d_mask_fill_holes           | core.cv2d_bmask    | Fill holes in connected objects in a given mask                           |
| cv2d_mask_close_contours       | core.cv2d_bmask    | Enclose objects with closely connected contours                           |
| cv2d_denoise_mask              | core.cv2d_bmask    | Fill holes, remove small objects                                          |
| cv2d_draw_bbox                 | core.cv2d_draw     | Draw bounding boxes on a given image with a specific color                |  
| cv2d_draw_boxes                | core.cv2d_draw     | Draw all bounding boxes on a given image with a specific color            |
| cv2d_draw_mask                 | core.cv2d_draw     | Overlay a mask on a given image with a specified color                    |
| cv2d_draw_mask_contour         | core.cv2d_draw     | Draw the contours of the mask on an image with a specified color          |
| cv2d_draw_masks_horizontal     | core.cv2d_draw     | Concatenate original image and one or multiple mask overlays              |
| cv2d_draw_error_map            | core.cv2d_draw     | Generate an error map for two binary masks                                |
| cv2d_detect_aruco_markers      | core.cv2d_marker   | Detect ARUCO markers on a given image                                     |
| cv2d_detect_aruco_markers_pose | core.cv2d_marker   | Detect ARUCO markers on a given image, estimate pose using cam parameters |  
| cv2d_swap_color                | core.cv2d_utils    | Swap one color in an image to another                                     |
| cv3d_binarize                  | core.cv3d_core     | Binarize a volume using the specified threshold                           |
| cv3d_connected_components      | core.cv3d_core     | Apply 3D connected components labelling to a binary volume                |


Scripts
- `run-labelme2mask.py` - generate masks from JSON files generated by labelme
  - `-i <path_to_dir>` - path to directory with JSON filse
  - `-o <path_to_dir>` - path to directory where generated masks will be stored

--------------------------------------------------------------------