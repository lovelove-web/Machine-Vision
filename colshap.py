from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


class ShapeColorDetector:
    COLOR_RANGES = {
        "red": [(0, 70, 50), (10, 255, 255), (165, 70, 50), (180, 255, 255)],
        "blue": [(95, 60, 50), (135, 255, 255)],
        "green": [(35, 30, 20), (90, 255, 255)],  # ← V lowered to 20 for dark greens
        "yellow": [(18, 80, 80), (35, 255, 255)],
        "orange": [(8, 100, 80), (18, 255, 255)],
        "purple": [
            (120, 30, 20),
            (165, 255, 200),
        ],  # ← Saturation lowered for dark purples
        "cyan": [(82, 60, 60), (98, 255, 255)],
        "black": [(0, 0, 0), (180, 80, 80)],
    }

    def __init__(self, min_area=500, max_area=100000):
        self.min_area = min_area
        self.max_area = max_area

    def _mask(self, hsv, color):
        r = self.COLOR_RANGES.get(color)
        if not r:
            return np.ones(hsv.shape[:2], np.uint8) * 255
        if color == "red":
            m = cv2.inRange(hsv, np.array(r[0]), np.array(r[1])) | cv2.inRange(
                hsv, np.array(r[2]), np.array(r[3])
            )
        else:
            m = cv2.inRange(hsv, np.array(r[0]), np.array(r[1]))
        k = np.ones((4, 4), np.uint8)
        return cv2.medianBlur(
            cv2.morphologyEx(
                cv2.morphologyEx(m, cv2.MORPH_CLOSE, k), cv2.MORPH_OPEN, k
            ),
            5,
        )

    def _shape(self, c):
        p = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * p, True)
        v, area = len(approx), cv2.contourArea(c)
        circ = 4 * np.pi * area / (p**2) if p > 0 else 0
        ar = (lambda b: b[2] / b[3] if b[3] > 0 else 0)(cv2.boundingRect(c))
        if circ > 0.82:
            return "circle"
        if v == 3:
            return "triangle"
        if v == 4:
            return "square" if 0.8 <= ar <= 1.2 else "rectangle"
        return "polygon"

    def detect(self, image, color_filter=None, shape_filter=None):
        blurred = cv2.GaussianBlur(image, (5, 5), 0)
        h, s, v = cv2.split(cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV))
        v = cv2.threshold(v, 252, 252, cv2.THRESH_TRUNC)[1]
        hsv = cv2.merge([h, s, v])

        colors = (
            [color_filter]
            if color_filter and color_filter != "any"
            else list(self.COLOR_RANGES.keys())
        )

        objects, centers = [], []
        for color in colors:
            for c in cv2.findContours(
                self._mask(hsv, color), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )[0]:
                area = cv2.contourArea(c)
                if not (self.min_area < area < self.max_area):
                    continue
                M = cv2.moments(c)
                if M["m00"] == 0:
                    continue
                cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
                if any(np.hypot(cx - x, cy - y) < 30 for x, y in centers):
                    continue
                bx, by, bw, bh = cv2.boundingRect(c)
                if bh == 0 or not (0.2 < bw / bh < 5.0):
                    continue
                ha = cv2.contourArea(cv2.convexHull(c))
                if ha == 0 or area / ha < 0.72:
                    continue  # ← relaxed for dark objects
                shape = self._shape(c)
                if shape_filter and shape_filter != "any" and shape != shape_filter:
                    continue
                objects.append(
                    {
                        "center": (cx, cy),
                        "color": color,
                        "shape": shape,
                        "area": area,
                        "contour": c,
                    }
                )
                centers.append((cx, cy))
        return objects

    def annotate(self, image, objects, show_labels=True):
        COLORS = {
            "red": (0, 0, 255),
            "blue": (255, 0, 0),
            "green": (0, 255, 0),
            "yellow": (0, 255, 255),
            "orange": (0, 165, 255),
            "purple": (255, 0, 255),
            "cyan": (255, 255, 0),
            "black": (50, 50, 50),
        }
        out = image.copy()
        for o in objects:
            cx, cy = o["center"]
            bgr = COLORS.get(o["color"], (128, 128, 128))
            cv2.drawContours(out, [o["contour"]], -1, bgr, 2)
            cv2.circle(out, (cx, cy), 5, (0, 255, 0), -1)
            cv2.line(out, (cx - 10, cy), (cx + 10, cy), (0, 255, 0), 2)
            cv2.line(out, (cx, cy - 10), (cx, cy + 10), (0, 255, 0), 2)
            if show_labels:
                cv2.putText(
                    out,
                    f"{o['shape']} ({o['color']})",
                    (cx, cy - 15),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    bgr,
                    2,
                )
        return out


if __name__ == "__main__":
    detector = ShapeColorDetector()
    image = cv2.imread("oko.jpeg")
    if image is not None:
        annotated = detector.annotate(image, detector.detect(image))
        cv2.imshow("Detected", annotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Image not found.")
