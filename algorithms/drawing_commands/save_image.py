import os

import cv2
import numpy as np


def main(gesture, coords, command_executor):
    if not command_executor.commands_variables.get('is_saved', False):
        command_executor.commands_variables['is_saved'] = True
        command_executor.commands_variables['image'] = command_executor.commands_variables.get('image',
                                                                                               np.zeros((480, 640)))
        cv2.imwrite(os.path.join('Data', 'train', 'test.jpg'), command_executor.commands_variables['image'])
        command_executor.commands_variables['image'] = np.zeros((480, 640))
        cv2.imshow('image', command_executor.commands_variables['image'])
