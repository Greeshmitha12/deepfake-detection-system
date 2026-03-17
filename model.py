import numpy as np
import cv2

def predict_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Blur
    blur = cv2.Laplacian(gray, cv2.CV_64F).var()

    # Edges
    edges = cv2.Canny(gray, 100, 200)
    edge_density = edges.mean()

    # Brightness
    brightness = gray.mean()

    # Noise
    noise = np.std(gray)

    # Normalize
    blur_score = max(0, min(1, 1 - (blur / 150)))
    edge_score = max(0, min(1, 1 - (edge_density / 50)))
    brightness_score = max(0, min(1, abs(brightness - 128) / 128))
    noise_score = max(0, min(1, noise / 50))

    # Final score
    score = (blur_score + edge_score + brightness_score + noise_score) / 4

    return score