import os
import time
import cv2
import numpy as np
import logging as log

from face_detection import FaceDetectionModel
from facial_landmarks_detection import FacialLandmarksDetectionModel
from gaze_estimation import GazeEstimationModel
from head_pose_estimation import HeadPoseEstimationModel
from mouse_controller import MouseController
from argparse import ArgumentParser
from input_feeder import InputFeeder

def build_argparser():
    """
    Parse command line arguments.
    :return: command line arguments
    """
    parser = ArgumentParser()
    parser.add_argument("-fd", "--face_detection", required=True, type=str,
                        help="Path to a face detection model xml file with a trained model.")
    parser.add_argument("-hp", "--head_pose_estimation", required=True, type=str,
                        help="Path to a head pose estimation model xml file with a trained model.")
    parser.add_argument("-fl", "--facial_landmarks_detection", required=True, type=str,
                        help="Path to a facial landmarks detection model xml file with a trained model.")
    parser.add_argument("-ge", "--gaze_estimation", required=True, type=str,
                        help="Path to a gaze estimation model xml file with a trained model.")
    parser.add_argument("-i", "--input", required=True, type=str,
                        help="Path to image or video file")
    parser.add_argument("-l", "--cpu_extension", required=False, type=str,
                        default=None,
                        help="MKLDNN (CPU)-targeted custom layers."
                             "Absolute path to a shared library with the"
                             "kernels impl.")
    parser.add_argument("-d", "--device", type=str, default="CPU",
                        help="Specify the target device to infer on: "
                             "CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
                             "will look for a suitable plugin for device "
                             "specified (CPU by default)")
    parser.add_argument("-pt", "--prob_threshold", type=float, default=0.5,
                        help="Probability threshold for detections filtering"
                        "(0.5 by default)")
    parser.add_argument("-flag", "--visualization_flag", required=False, nargs='+',
                        default=[],
                        help="Specify the flags from fd hp fl ge, like --flag fd hp fl ge (Seperate each flag by space)"
                             "for see the visualization of different model outputs of each frame,"
                             "fd for Face Detection Model, hp for Head Pose Estimation Model"
                             "fl for Facial Landmark Detection Model, ge for Gaze Estimation Model.")
    return parser


def infer_on_stream(args):
    
    input_file_path = args.input # Path to image or video file
    logger_object = log.getLogger()
    visualization_flag = args.visualization_flag
    
    input_feeder = input_feeder(input_file_path)
    
    face_detection_instant, head_pose_estimation_instant, facial_landmarks_instant, gaze_estimation_instant = \
    model_instants()

#     # Checks for live feed
#     if input_file_path == 'CAM':
#         input_feeder = InputFeeder("cam")

#     # Checks for video file
#     else:
#         input_feeder = InputFeeder("video", input_file_path)
#         assert os.path.isfile(input_file_path), "Specified input file doesn't exist"
        
        
#     model_paths_dict = {'face_detection':args.face_detection,
#                         'head_pose_estimation':args.head_pose_estimation,
#                         'facial_landmarks_detection':args.facial_landmarks_detection,
#                         'gaze_estimation':args.gaze_estimation}

#     face_detection_instant = FaceDetectionModel(model_name=args.face_detection,
#                                                 device=args.device, threshold=args.prob_threshold,
#                                                 extensions=args.cpu_extension)
    
#     head_pose_estimation_instant = HeadPoseEstimationModel(model_name=args.head_pose_estimation,
#                                                            device=args.device, 
#                                                            extensions=args.cpu_extension)

#     facial_landmarks_instant = FacialLandmarksDetectionModel(model_name=args.facial_landmarks_detection,
#                                                              device=args.device,
#                                                              extensions=args.cpu_extension)

#     gaze_estimation_instant = GazeEstimationModel(model_name=args.gaze_estimation, 
#                                                   device=args.device, 
#                                                   extensions=args.cpu_extension)
    
#     mouse_controller_instant = MouseController('medium', 'fast')


    face_detection_instant.load_model()
    
    head_pose_estimation_instant.load_model()
    
    facial_landmarks_instant.load_model()
    
    gaze_estimation_instant.load_model()
    
    input_feeder.load_data()
        
        
        
        
    
    
    
    


    cap = cv2.VideoCapture(input_stream)
    if input_stream:
        cap.open(args.input)

    if not cap.isOpened():
        log.error("ERROR! Unable to open video source")

    global initial_w, initial_h
    #prob_threshold = args.prob_threshold
    initial_w = cap.get(3)
    initial_h = cap.get(4)

    # Flag for the input image
    single_image_mode = False

    #iniatilize desired variables
    global total_count, report, duration, net_output, frame, prev_counter, prev_duration, counter, dur
    
    total_count = 0
    request_id=0
    
    report = 0


    prev_counter = 0
    prev_duration = 0
    counter = 0
    dur = 0


    # NOTE: Some code implementation gotten from 
    # https://github.com/prateeksawhney97/People-Counter-Application-Using-Intel-OpenVINO-Toolkit/blob/master/main.py

    ### TODO: Loop until stream is over ###
    while cap.isOpened():

        ### TODO: Read from the video capture ###
        flag, frame = cap.read()
        if not flag:
            break
        key_pressed = cv2.waitKey(60)

        ### TODO: Pre-process the image as needed ###
        p_frame = cv2.resize(frame, (input_shape[3], input_shape[2]))
        p_frame = p_frame.transpose((2,0,1))
        p_frame = p_frame.reshape(1, *p_frame.shape)

        ### TODO: Start asynchronous inference for specified request ###
        net_input = {'image_tensor': p_frame,'image_info': p_frame.shape[1:]}
        duration = None
        infer_network.exec_net(request_id, net_input)

        ### TODO: Wait for the result ###
        if infer_network.wait(request_id) == 0:

            ### TODO: Get the results of the inference request ###
            net_output = infer_network.get_output(request_id)

            ### TODO: Extract any desired stats from the results ###
            #probs = net_output[0, 0, :, 2]

            frame, net_output, prev_counter, prev_duration, counter, dur, report, total_count, duration = \
            desired_stats(frame, net_output, prev_counter, 
                prev_duration, counter, dur, report, total_count, duration)

            ### TODO: Calculate and send relevant information on ###
            ### current_count, total_count and duration to the MQTT server ###
            ### Topic "person": keys of "count" and "total" ###
            ### Topic "person/duration": key of "duration" ###

            # When new person enters the video
            client.publish('person',
               json.dumps({
                   'count': report, 'total': total_count}),
               qos=0, retain=False)

            # Person duration in the video is calculated
            if duration is not None:
                client.publish('person/duration', 
                    json.dumps({'duration': duration}), 
                    qos=0, retain=False)


            if key_pressed == 27:
                break

        ### TODO: Send the frame to the FFMPEG server ###
        frame = cv2.resize(frame, (768, 432))
        sys.stdout.buffer.write(frame)
        sys.stdout.flush()

        ### TODO: Write an output image if `single_image_mode` ###
        if single_image_mode:
            cv2.imwrite('output_image.jpg', frame)

    cap.release()
    cv2.destroyAllWindows()
    client.disconnect()
    infer_network.clean()
    
    
def input_feeder(input_file_path):
    # Checks for live feed
    if input_file_path == 'CAM':
        input_feeder = InputFeeder("cam")

    # Checks for video file
    else:
        input_feeder = InputFeeder("video", input_file_path)
        assert os.path.isfile(input_file_path), "Specified input file doesn't exist"
        
    return input_feeder

def model_instants():
#     model_paths_dict = {'face_detection':args.face_detection,
#                         'head_pose_estimation':args.head_pose_estimation,
#                         'facial_landmarks_detection':args.facial_landmarks_detection,
#                         'gaze_estimation':args.gaze_estimation}

    face_detection_instant = FaceDetectionModel(model_name=args.face_detection,
                                                device=args.device, threshold=args.prob_threshold,
                                                extensions=args.cpu_extension)
    
    head_pose_estimation_instant = HeadPoseEstimationModel(model_name=args.head_pose_estimation,
                                                           device=args.device, 
                                                           extensions=args.cpu_extension)

    facial_landmarks_instant = FacialLandmarksDetectionModel(model_name=args.facial_landmarks_detection,
                                                             device=args.device,
                                                             extensions=args.cpu_extension)

    gaze_estimation_instant = GazeEstimationModel(model_name=args.gaze_estimation, 
                                                  device=args.device, 
                                                  extensions=args.cpu_extension)
    
    mouse_controller_instant = MouseController('medium', 'fast')
    
    return face_detection_instant, head_pose_estimation_instant, facial_landmarks_instant, gaze_estimation_instant 

def main():
    """
    Load the network and parse the output.
    :return: None
    """
    # Grab command line args
    args = build_argparser().parse_args()

    # Perform inference on the input stream
    infer_on_stream(args)


if __name__ == '__main__':
    main()