# core-common-tools
Common tools and helper scripts

--------------------------------------------------------------------

**cv_common_utils**

| Function name              | Package           | Description                                                             |
|----------------------------|-------------------|-------------------------------------------------------------------------|
| cv2d_convert_bbox2mask     | core.cv2d_convert | Generate mask paining given list of bounding boxes with intensity score |
| cv2d_convert_mask2bbox     | core.cv2d_convert | Calculate bounding boxes of connected components in a binary mask       |
| cv2d_convert_polygon2mask  | core.cv2d_convert | Generate masks painting given polygon wih specified intensity           |
| cv2d_convert_polygons2mask | core.cv2d_convert | Generate masks painting given list polygons wih specified intensity     |
| cv2d_convert_mask2polygon  | core.cv2d_convert | Calculate polygons for connected components of a mask                   |
| cv2d_convert_pix2label     | core.cv2d_convert | Convert RGB image to label map using the given LUT                      |
| cv2d_convert_label2pix     | core.cv2d_convert | Convert label map to RGB image using the given LUT                      |
| cv2d_mask_fill_holes       | core.cv2d_bmask   | Fill holes in connected objects in a given mask                         |
| cv2d_mask_close_contours   | core.cv2d_bmask   | Enclose objects with closely connected contours                         |
| cv2d_denoise_mask          | core.cv2d_bmask   | Fill holes, remove small objects                                        |
| cv2d_draw_bbox             | core.cv2d_draw    | Draw bounding boxes on a given image with a specific color              |  
| cv2d_draw_boxes            | core.cv2d_draw    |
| cv2d_draw_mask             | core.cv2d_draw    |
| cv2d_draw_mask_contour     | core.cv2d_draw    |
| cv2d_draw_masks_horizontal | core.cv2d_draw    |
| cv2d_draw_error_map        | core.cv2d_draw    |