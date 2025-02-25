import os

import cv2
import numpy as np
from PIL import Image


def main(gesture, coords, command_executor):
    if not command_executor.commands_variables.get('is_saved', False):
        command_executor.commands_variables['is_saved'] = True
        command_executor.commands_variables['image'] = command_executor.commands_variables.get('image',
                                                                                               np.zeros((480, 640)))
        print("Текст распознаётся. Ожидайте...")
        cv2.imwrite('test.jpg', command_executor.commands_variables['image'])
        image = Image.open("test.jpg").convert("RGB")
        pixel_values = command_executor.commands_variables['processor'](images=image, return_tensors="pt").pixel_values

        generated_ids = command_executor.commands_variables['model'].generate(pixel_values)
        generated_text = command_executor.commands_variables['processor'].batch_decode(generated_ids,
                                                                                       skip_special_tokens=True)[0]
        print("Результат распознавания: ", generated_text)

        command_executor.commands_variables['image'] = np.zeros((480, 640))
        cv2.imshow('image', command_executor.commands_variables['image'])