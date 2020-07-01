'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''

class Model_X:
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
            if cpu_extension and "CPU" in device:
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
        
        face_coordinates = preprocess_output(outputs, image)
        if len(face_coordinates) == 0:
            log.error("No face detected")
            return 0, 0
        
        first_detected_face = face_coordinates[0]
        cropped_face = image[face_coordinates[1]:face_coordinates[3], face_coordinates[0]:face_coordinates[2]]
        
        return first_detected_face, cropped_face

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
        
        face_coordinates = []
        net_output = outputs[self.output_name][0][0]
        
        for obj in net_output:
            if obj[2] > self.threshold:
                x_min = int(obj[3] * initial_w)
                y_min = int(obj[4] * initial_h)
                x_max = int(obj[5] * initial_w)
                y_max = int(obj[6] * initial_h)
                
                face_coordinates.append([x_min, y_min, x_max, y_max])

        return face_coordinates
