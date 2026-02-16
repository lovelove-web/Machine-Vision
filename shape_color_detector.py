import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict


class ShapeColorDetector:
    """Detects objects by shape and color with enhanced filtering."""

    # Color ranges in HSV - more restrictive to avoid background
    COLOR_RANGES = {
        'red': [(0, 120, 100), (10, 255, 255), (170, 120, 100), (180, 255, 255)],  # Red wraps around
        'blue': [(100, 120, 70), (130, 255, 255)],
        'green': [(35, 50, 50), (85, 255, 255)],  # Wider range for green
        'yellow': [(20, 100, 100), (35, 255, 255)],  # Lower saturation for yellow
        'orange': [(10, 120, 100), (20, 255, 255)],
        'purple': [(130, 80, 70), (160, 255, 255)],
        'cyan': [(80, 120, 100), (100, 255, 255)],
        'black': [(0, 0, 0), (180, 50, 80)],  # Narrow value range for black only
    }

    SHAPE_TYPES = ['circle', 'square', 'rectangle', 'triangle', 'any']

    def __init__(self):
        self.min_area = 1000  # Minimum contour area - increased to filter small objects
        self.max_area = 100000  # Maximum contour area

    def detect_objects(self, image: np.ndarray,
                      color_filter: Optional[str] = None,
                      shape_filter: Optional[str] = None) -> List[Dict]:
        """
        Detect objects in image with optional color and shape filtering.

        Args:
            image: Input BGR image
            color_filter: Color to filter ('red', 'blue', etc., or None for any)
            shape_filter: Shape to filter ('circle', 'square', etc., or None for any)

        Returns:
            List of detected objects with properties:
            - center: (x, y) pixel coordinates
            - color: detected color name
            - shape: detected shape name
            - area: contour area
            - contour: contour points
        """
        objects = []
        detected_centers = []  # Track centers to avoid duplicates

        # Convert to HSV for color detection
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Determine which colors to check
        # Order matters - check specific colors before black to avoid misclassification
        if color_filter and color_filter != 'any':
            colors_to_check = [color_filter]
        else:
            # Check in priority order - specific colors first, then black
            colors_to_check = ['red', 'blue', 'green', 'yellow', 'orange', 'purple', 'cyan', 'black']

        for color_name in colors_to_check:
            # Create color mask
            mask = self._create_color_mask(hsv, color_name)

            # Find contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                area = cv2.contourArea(contour)

                # Filter by area
                if area < self.min_area or area > self.max_area:
                    continue

                # Get center point first to check for duplicates
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                else:
                    continue

                # Check if this object was already detected (within 30 pixels)
                is_duplicate = False
                for prev_center in detected_centers:
                    dist = np.sqrt((cx - prev_center[0])**2 + (cy - prev_center[1])**2)
                    if dist < 30:
                        is_duplicate = True
                        break

                if is_duplicate:
                    continue

                # Get bounding box to filter out very elongated objects (likely edges/lines)
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = float(w) / h if h > 0 else 0

                # Skip very elongated objects (likely background lines/edges)
                if aspect_ratio < 0.2 or aspect_ratio > 5.0:
                    continue

                # Check if object has enough pixels (not just thin outline)
                bbox_area = w * h
                if bbox_area > 0 and (area / bbox_area) < 0.3:
                    continue

                # Get shape properties
                shape_name, properties = self._classify_shape(contour)

                # Apply shape filter
                if shape_filter and shape_filter != 'any' and shape_name != shape_filter:
                    continue

                # Store object information
                obj = {
                    'center': (cx, cy),
                    'color': color_name,
                    'shape': shape_name,
                    'area': area,
                    'contour': contour,
                    'properties': properties
                }

                objects.append(obj)
                detected_centers.append((cx, cy))

        return objects

    def _create_color_mask(self, hsv_image: np.ndarray, color_name: str) -> np.ndarray:
        """Create binary mask for specified color."""
        if color_name not in self.COLOR_RANGES:
            # Return full white mask if color not recognized
            return np.ones(hsv_image.shape[:2], dtype=np.uint8) * 255

        ranges = self.COLOR_RANGES[color_name]

        # Handle red color (wraps around HSV hue)
        if color_name == 'red':
            lower1, upper1, lower2, upper2 = ranges
            mask1 = cv2.inRange(hsv_image, np.array(lower1), np.array(upper1))
            mask2 = cv2.inRange(hsv_image, np.array(lower2), np.array(upper2))
            mask = cv2.bitwise_or(mask1, mask2)
        else:
            lower, upper = ranges
            mask = cv2.inRange(hsv_image, np.array(lower), np.array(upper))

        # Morphological operations to reduce noise - stronger filtering
        kernel = np.ones((7, 7), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        # Remove small noise
        kernel_small = np.ones((3, 3), np.uint8)
        mask = cv2.erode(mask, kernel_small, iterations=1)
        mask = cv2.dilate(mask, kernel_small, iterations=1)

        return mask

    def _classify_shape(self, contour: np.ndarray) -> Tuple[str, Dict]:
        """
        Classify shape based on contour properties.

        Returns:
            tuple: (shape_name, properties_dict)
        """
        properties = {}

        # Approximate contour to polygon
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)  # More precise approximation
        num_vertices = len(approx)

        properties['vertices'] = num_vertices
        properties['perimeter'] = perimeter

        # Calculate circularity
        area = cv2.contourArea(contour)
        if perimeter > 0:
            circularity = 4 * np.pi * area / (perimeter ** 2)
        else:
            circularity = 0

        properties['circularity'] = circularity

        # Fit bounding rectangle
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = float(w) / h if h > 0 else 0
        properties['aspect_ratio'] = aspect_ratio

        # Classify shape - prioritize circularity over vertex count
        if circularity > 0.85:
            shape = 'circle'
        elif num_vertices == 3:
            shape = 'triangle'
        elif num_vertices == 4:
            # Distinguish between square and rectangle
            if 0.85 <= aspect_ratio <= 1.15:
                shape = 'square'
            else:
                shape = 'rectangle'
        elif num_vertices == 5:
            shape = 'pentagon'
        elif num_vertices == 6:
            shape = 'hexagon'
        elif num_vertices > 6 and circularity > 0.7:
            # Many vertices but still round - likely a circle
            shape = 'circle'
        else:
            shape = 'polygon'

        return shape, properties

    def detect_circles(self, image: np.ndarray, color_filter: Optional[str] = None) -> List[Dict]:
        """
        Enhanced circle detection using blob detector (based on original code).

        Args:
            image: Input BGR image
            color_filter: Optional color filter

        Returns:
            List of detected circular objects
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply color filter if specified
        if color_filter and color_filter != 'any':
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            mask = self._create_color_mask(hsv, color_filter)
            gray = cv2.bitwise_and(gray, gray, mask=mask)

        # Threshold to get binary image
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

        # Setup blob detector parameters (from original code)
        params = cv2.SimpleBlobDetector_Params()
        params.blobColor = 255

        # Filter by circularity
        params.filterByCircularity = True
        params.minCircularity = 0.8

        # Filter by area
        params.filterByArea = True
        params.minArea = self.min_area
        params.maxArea = self.max_area

        # Create detector
        detector = cv2.SimpleBlobDetector_create(params)

        # Detect blobs
        keypoints = detector.detect(binary)

        # Convert keypoints to object format
        objects = []
        for kp in keypoints:
            obj = {
                'center': (int(kp.pt[0]), int(kp.pt[1])),
                'color': color_filter if color_filter else 'unknown',
                'shape': 'circle',
                'area': np.pi * (kp.size / 2) ** 2,
                'radius': kp.size / 2
            }
            objects.append(obj)

        return objects

    def annotate_image(self, image: np.ndarray, objects: List[Dict],
                       show_labels: bool = True) -> np.ndarray:
        """
        Draw detected objects on image with annotations.

        Args:
            image: Input image
            objects: List of detected objects
            show_labels: Whether to show text labels

        Returns:
            Annotated image
        """
        annotated = image.copy()

        for i, obj in enumerate(objects):
            cx, cy = obj['center']
            color = obj.get('color', 'unknown')
            shape = obj.get('shape', 'unknown')

            # Choose annotation color based on detected color
            color_bgr = self._get_display_color(color)

            # Draw contour if available
            if 'contour' in obj:
                cv2.drawContours(annotated, [obj['contour']], -1, color_bgr, 2)

            # Draw center point
            cv2.circle(annotated, (cx, cy), 5, (0, 255, 0), -1)

            # Draw crosshair
            cv2.line(annotated, (cx - 10, cy), (cx + 10, cy), (0, 255, 0), 2)
            cv2.line(annotated, (cx, cy - 10), (cx, cy + 10), (0, 255, 0), 2)

            if show_labels:
                # Create label
                label = f"{shape} ({color})"

                # Position label
                label_y = cy - 20

                # Draw label background
                (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                cv2.rectangle(annotated, (cx - 5, label_y - label_h - 5),
                            (cx + label_w + 5, label_y + 5), (255, 255, 255), -1)

                # Draw label text
                cv2.putText(annotated, label, (cx, label_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_bgr, 2)

        return annotated

    def _get_display_color(self, color_name: str) -> Tuple[int, int, int]:
        """Get BGR color for display based on color name."""
        color_map = {
            'red': (0, 0, 255),
            'blue': (255, 0, 0),
            'green': (0, 255, 0),
            'yellow': (0, 255, 255),
            'orange': (0, 165, 255),
            'purple': (255, 0, 255),
            'cyan': (255, 255, 0),
            'white': (255, 255, 255),
            'black': (0, 0, 0),
        }
        return color_map.get(color_name, (128, 128, 128))


# Testing function
if __name__ == "__main__":
    # Test the detector
    detector = ShapeColorDetector()

    # Try to load test image
    test_image_path = 'imgtry1.jpg'
    image = cv2.imread(test_image_path)

    if image is not None:
        print("Testing shape and color detection...")

        # Detect all objects
        objects = detector.detect_objects(image)
        print(f"\nDetected {len(objects)} objects:")

        for i, obj in enumerate(objects):
            print(f"{i+1}. Shape: {obj['shape']}, Color: {obj['color']}, "
                  f"Center: {obj['center']}, Area: {obj['area']:.0f}")

        # Annotate and display
        annotated = detector.annotate_image(image, objects)
        cv2.imshow('Detected Objects', annotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print(f"Could not load image: {test_image_path}")
        print("Place a test image named 'rbdk_image.png' in the current directory")
