'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''

import cv2
import os
import numpy as np
import logging as log
import math
from openvino.inference_engine import IENetwork, IECore

class GazeEstimationModel:
    '''
    Class for the Face Detection Model.
    '''
    def __init__(self, model_name, device='CPU', threshold=0.60, extensions=None):
        '''
        COMPLETED: Use this to set your instance variables.
        '''
        self.model_weights=model_name+'.bin'
        self.model_structure=model_name+'.xml'
        self.device=device
        self.threshold=threshold
        self.extension = extensions
        self.net = None

        try:
            self.model=IENetwork(self.model_structure, self.model_weights)
        except Exception as e:
            raise ValueError("Could not Initialise the network. Have you enterred the correct model path?")

        self.input_name=next(iter(self.model.inputs))
        self.input_shape=self.model.inputs[self.input_name].shape
        self.output_name=next(iter(self.model.outputs))
        self.output_shape=self.model.outputs[self.output_name].shape

    def load_model(self):
        '''
        COMPLETED: You will need to complete this method.
        This method is for loading the model to the device specified by the user.
        If your model requires any Plugins, this is where you can load them.
        '''
        self.core = IECore()
        
        #get supported layers
        supported_layers = self.core.query_network(self.model, self.device)

        # Check for any unsupported layers, and let the user
        #  know if anything is missing. Exit the program, if so.
        unsupported_layers = [l for l in self.model.layers.keys() if l not in supported_layers]
        if len(unsupported_layers) != 0:
            log.error("Unsupported layers found: {}".format(unsupported_layers))
            log.error("Check whether extensions are available to add to IECore.")
            # exit(1)
            # Add a CPU extension, if applicable
            if self.extension and "CPU" in self.device:
                self.core.add_extension(self.extension, self.device)
        
        self.net = self.core.load_network(network=self.model, device_name=self.device, num_requests=1)
        
        return

    def predict(self, left_eye_image, right_eye_image, head_pose_angles):
        '''
        COMPLETED: You will need to complete this method.
        This method is meant for running predictions on the input image.
        '''
        left_eye_input_img, right_eye_input_img = \
        self.preprocess_input(left_eye_image, right_eye_image)
        
        input_dict={'left_eye_image':left_eye_input_img, 'right_eye_image':right_eye_input_img, 
                    'head_pose_angles':head_pose_angles}        
        outputs = self.net.infer(input_dict)
        
        new_mouse_coords, gaze_vector = self.preprocess_output(outputs, head_pose_angles)
        
        return new_mouse_coords, gaze_vector

    def check_model(self):
        pass

    def preprocess_input(self, left_eye, right_eye):
        '''
        Before feeding the data into the model for inference,
        you might have to preprocess it. This function is where you can do that.
        '''
        left_eye_p_frame = cv2.resize(left_eye, (60, 60))
        left_eye_p_frame = left_eye_p_frame.transpose((2,0,1))
        left_eye_p_frame = left_eye_p_frame.reshape(1, *left_eye_p_frame.shape) # left eye pre frame
        
        right_eye_p_frame = cv2.resize(right_eye, (60, 60))
        right_eye_p_frame = right_eye_p_frame.transpose((2,0,1))
        right_eye_p_frame = right_eye_p_frame.reshape(1, *right_eye_p_frame.shape) # right eye pre frame
        
        return left_eye_p_frame, right_eye_p_frame

    def preprocess_output(self, outputs, head_pose_angles):
        '''
        Before feeding the output of this model to the next model,
        you might have to preprocess the output. This function is where you can do that.
        '''
        gaze_vector = outputs[self.output_name][0] # net_output
        roll_value = head_pose_angles[2]
        
        cos = math.cos(roll_value * math.pi / 180.0)
        sin = math.sin(roll_value * math.pi / 180.0)
        
        new_x_value = gaze_vector[0] * cos + gaze_vector[1] * sin
        new_y_value = gaze_vector[1] * cos - gaze_vector[0] * sin

        return (new_x_value, new_y_value), gaze_vector
