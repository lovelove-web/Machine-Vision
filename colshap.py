import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict

class ShapeColorDetector:
    # HSV color ranges
    COLOR_RANGES = {
        'red': [(0, 120, 100), (10, 255, 255), (170, 120, 100), (180, 255, 255)],
        'blue': [(100, 120, 70), (130, 255, 255)], 'green': [(35, 50, 50), (85, 255, 255)],
        'yellow': [(20, 100, 100), (35, 255, 255)], 'orange': [(10, 120, 100), (20, 255, 255)],
        'purple': [(130, 80, 70), (160, 255, 255)], 'cyan': [(80, 120, 100), (100, 255, 255)],
        'black': [(0, 0, 0), (180, 50, 80)],
    }

    def __init__(self):
        self.min_area = 1000
        self.max_area = 100000

    def detect_objects(self, image: np.ndarray,
                       color_filter: Optional[str] = None,
                       shape_filter: Optional[str] = None) -> List[Dict]:
        objects, detected_centers = [], []
        # --- Reduce reflections ---
        blurred = cv2.GaussianBlur(image, (7, 7), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

        # Suppress very bright reflection pixels
        h, s, v = cv2.split(hsv)
        v = cv2.threshold(v, 240, 240, cv2.THRESH_TRUNC)[1]
        hsv = cv2.merge([h, s, v])
        # Select colors
        colors = [color_filter] if color_filter and color_filter != 'any' else \
                 ['red', 'blue', 'green', 'yellow', 'orange', 'purple', 'cyan', 'black']

        for color_name in colors:
            mask = self._create_color_mask(hsv, color_name)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                area = cv2.contourArea(contour)
                if not (self.min_area < area < self.max_area):
                    continue

                M = cv2.moments(contour)
                if M["m00"] == 0:
                    continue
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])

                # Avoid duplicate detections
                if any(np.hypot(cx - x, cy - y) < 30 for x, y in detected_centers):
                    continue
                x, y, w, h = cv2.boundingRect(contour)
                if h == 0 or not (0.2 < w / h < 5.0):
                    continue
                # Strong solidity check (reflection resistant)
                hull = cv2.convexHull(contour)
                hull_area = cv2.contourArea(hull)
                if hull_area == 0 or area / hull_area < 0.85:
                    continue
                shape = self._classify_shape(contour)
                if shape_filter and shape_filter != 'any' and shape != shape_filter:
                    continue

                objects.append({
                    'center': (cx, cy), 'color': color_name, 'shape': shape,
                    'area': area, 'contour': contour
                })
                detected_centers.append((cx, cy))
        return objects

    def _create_color_mask(self, hsv: np.ndarray, color: str) -> np.ndarray:
        if color not in self.COLOR_RANGES:
            return np.ones(hsv.shape[:2], dtype=np.uint8) * 255

        ranges = self.COLOR_RANGES[color]

        # Handle red wrap-around
        if color == 'red':
            l1, u1, l2, u2 = ranges
            mask = cv2.inRange(hsv, np.array(l1), np.array(u1)) | \
                   cv2.inRange(hsv, np.array(l2), np.array(u2))
        else:
            lower, upper = ranges
            mask = cv2.inRange(hsv, np.array(lower), np.array(upper))

        # Noise removal
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.medianBlur(mask, 5)

        return mask

    def _classify_shape(self, contour: np.ndarray) -> str:
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        vertices = len(approx)

        area = cv2.contourArea(contour)
        circularity = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0

        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / h if h > 0 else 0

        # Shape classification
        if circularity > 0.85:
            return 'circle'
        if vertices == 3:
            return 'triangle'
        if vertices == 4:
            return 'square' if 0.8 <= aspect_ratio <= 1.2 else 'rectangle'
        return 'polygon'

    def annotate_image(self, image: np.ndarray,
                       objects: List[Dict],
                       show_labels: bool = True) -> np.ndarray:

        annotated = image.copy()

        for obj in objects:
            cx, cy = obj['center']
            color_bgr = self._get_display_color(obj.get('color', 'unknown'))

            if 'contour' in obj:
                cv2.drawContours(annotated, [obj['contour']], -1, color_bgr, 2)

            # Draw center + crosshair
            cv2.circle(annotated, (cx, cy), 5, (0, 255, 0), -1)
            cv2.line(annotated, (cx - 10, cy), (cx + 10, cy), (0, 255, 0), 2)
            cv2.line(annotated, (cx, cy - 10), (cx, cy + 10), (0, 255, 0), 2)

            if show_labels:
                label = f"{obj['shape']} ({obj['color']})"
                cv2.putText(annotated, label, (cx, cy - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_bgr, 2)

        return annotated

    def _get_display_color(self, name: str) -> Tuple[int, int, int]:
        return {
            'red': (0, 0, 255), 'blue': (255, 0, 0), 'green': (0, 255, 0),
            'yellow': (0, 255, 255), 'orange': (0, 165, 255), 'purple': (255, 0, 255),
            'cyan': (255, 255, 0), 'black': (0, 0, 0),
        }.get(name, (128, 128, 128))

if __name__ == "__main__":
    detector = ShapeColorDetector()
    image = cv2.imread("imgtry1.jpg")

    if image is not None:
        objects = detector.detect_objects(image)
        annotated = detector.annotate_image(image, objects)
        cv2.imshow("Detected Objects", annotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Image not found.")
