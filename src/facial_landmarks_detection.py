'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''

import cv2
import os
import numpy as np
import logging as log
from openvino.inference_engine import IENetwork, IECore

class FacialLandmarksDetectionModel:
    '''
    Class for the Face Detection Model.
    '''
    def __init__(self, model_name, device='CPU', extensions=None):
        '''
        COMPLETED: Use this to set your instance variables.
        '''
        self.model_weights=model_name+'.bin'
        self.model_structure=model_name+'.xml'
        self.device=device
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

    def predict(self, image):
        '''
        COMPLETED: You will need to complete this method.
        This method is meant for running predictions on the input image.
        '''
        input_img = self.preprocess_input(image)
        input_dict={self.input_name:input_img}        
        outputs = self.net.infer(input_dict)
        
        left_eye_x_coord, left_eye_y_coord, right_eye_x_coord, right_eye_y_coord = \
        self.preprocess_output(outputs, image)
        
        left_eye_x_min_coord = left_eye_x_coord - 10
        left_eye_y_min_coord = left_eye_y_coord - 10
        left_eye_x_max_coord = left_eye_x_coord + 10
        left_eye_y_max_coord = left_eye_y_coord + 10
        
        right_eye_x_min_coord = right_eye_x_coord - 10
        right_eye_y_min_coord = right_eye_y_coord - 10
        right_eye_x_max_coord = right_eye_x_coord + 10
        right_eye_y_max_coord = right_eye_y_coord + 10
        
        left_eye = image[left_eye_x_min_coord:left_eye_x_max_coord, left_eye_y_min_coord:left_eye_y_max_coord]
        right_eye = image[right_eye_x_min_coord:right_eye_x_max_coord, right_eye_y_min_coord:right_eye_y_max_coord]
        
        eye_coords = [[left_eye_x_min_coord, left_eye_y_min_coord, left_eye_x_max_coord, left_eye_y_max_coord], 
                     [right_eye_x_min_coord, right_eye_y_min_coord, right_eye_x_max_coord, right_eye_y_max_coord]]
        
        return left_eye, right_eye, eye_coords

    def check_model(self):
        pass

    def preprocess_input(self, image):
        '''
        Before feeding the data into the model for inference,
        you might have to preprocess it. This function is where you can do that.
        '''
        p_frame = cv2.resize(image, (self.input_shape[3], self.input_shape[2]))
        p_frame = p_frame.transpose((2,0,1))
        p_frame = p_frame.reshape(1, *p_frame.shape)
        
        return p_frame

    def preprocess_output(self, outputs, image):
        '''
        Before feeding the output of this model to the next model,
        you might have to preprocess the output. This function is where you can do that.
        '''
        initial_w = image.shape[1] # image width
        initial_h = image.shape[0] # image height
        
        outputs = outputs[self.output_name][0]
        
        left_eye_x_coord = int(outputs[0] * initial_w)
        left_eye_y_coord = int(outputs[1] * initial_h)
        right_eye_x_coord = int(outputs[2] * initial_w)
        right_eye_y_coord = int(outputs[3] * initial_h)

        return left_eye_x_coord, left_eye_y_coord, right_eye_x_coord, right_eye_y_coord
