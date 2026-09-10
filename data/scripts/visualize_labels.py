"""data/scripts/visualize_labels.py - Visual verification of ground truth bounding boxes."""

# ==============================================================================
# 1. IMPORTS & DEPENDENCIES
# ==============================================================================
# [FROM: Python Standard Library] pathlib.Path for paths
from pathlib import Path
# [FROM: OpenCV] cv2 for reading and drawing boxes on image
import cv2

# ==============================================================================
# 2. IMAGE & LABEL VISUALIZATION
# ==============================================================================
# [DEF: Paths] Sample image and corresponding YOLO annotation file
image_path = Path("data/processed/train/images/frame_0123.jpg")
label_path = Path("data/processed/labels/frame_0123.txt")

# [FUNCTION: Load Image]
image = cv2.imread(str(image_path))

if image is not None and label_path.exists():
    height, width = image.shape[:2]

    # [FUNCTION: Parse & Draw Bounding Boxes]
    with open(label_path, "r") as file:
        for line in file:
            class_id, x_center, y_center, box_width, box_height = map(
                float, line.split()
            )

            # Convert normalized coordinates back to pixel coordinates
            x_center *= width
            y_center *= height
            box_width *= width
            box_height *= height

            x1 = int(x_center - box_width / 2)
            y1 = int(y_center - box_height / 2)
            x2 = int(x_center + box_width / 2)
            y2 = int(y_center + box_height / 2)

            # Draw green bounding box
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.imshow("PETS-2009 Ground Truth", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print(f"Sample file not found: {image_path} or {label_path}")
