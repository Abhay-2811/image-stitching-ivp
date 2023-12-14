import cv2
import numpy as np

def resize_images(images, target_width):
    return [cv2.resize(img, (target_width, int(target_width / img.shape[1] * img.shape[0])), interpolation=cv2.INTER_AREA) for img in images]

def vertical_stitch(images):
    # Ensure that all images have the same width
    target_width = max(img.shape[1] for img in images)
    
    # Resize images to have the same width
    resized_images = resize_images(images, target_width)
    
    # Initialize an empty canvas with a height sufficient to accommodate all images
    total_height = sum(img.shape[0] for img in resized_images)
    result = np.zeros((total_height, target_width, 3), dtype=np.uint8)
    
    # Paste each resized image onto the canvas vertically
    y_offset = 0
    for img in resized_images:
        result[y_offset:y_offset+img.shape[0], :target_width, :] = img
        y_offset += img.shape[0]
    
    return result

image_paths = ['v1.jpg', 'v2.jpg', 'v3.jpg']

# Read images
images = [cv2.imread(img_path) for img_path in image_paths]

# Perform vertical stitching with resizing
result_image = vertical_stitch(images)

# Display or save the result image
cv2.imshow('Vertical Stitching', result_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Save the result image
cv2.imwrite('vertical_result.jpg', result_image)
