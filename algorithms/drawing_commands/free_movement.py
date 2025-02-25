import cv2
import numpy as np
from transformers import TrOCRProcessor, VisionEncoderDecoderModel


def main(gesture, coords, command_executor):
    if command_executor.commands_variables.get('processor') is None:
        print("Модель загружается...")
        command_executor.commands_variables['processor'] = TrOCRProcessor.from_pretrained("trained_model/weights")
        command_executor.commands_variables['model'] = VisionEncoderDecoderModel.from_pretrained("trained_model/weights")
        print("Модель загружена.")
    command_executor.commands_variables['is_saved'] = False
    command_executor.commands_variables['line_coords'] = []
    command_executor.commands_variables['image'] = command_executor.commands_variables.get('image',
                                                                                           np.zeros((480, 640)))
    image_width = 640 - float(coords[2]) - 10
    image_height = 480 - float(coords[3]) - 10
    x_norm = float(coords[0] - coords[2] / 2) / image_width
    y_norm = float(coords[1] - coords[3] / 2) / image_height
    if x_norm > 1.0:
        x_norm = 1.0
    if x_norm < 0.0:
        x_norm = 0.0
    if y_norm > 1.0:
        y_norm = 1.0
    if y_norm < 0.0:
        y_norm = 0.0
    x = int(x_norm * image_width)
    y = int(y_norm * image_height)
    image = command_executor.commands_variables['image'].copy()
    cv2.circle(image,
               (x, y),
               radius=2,
               color=(255, 0, 0),
               thickness=-1)
    cv2.imshow('image', image)
