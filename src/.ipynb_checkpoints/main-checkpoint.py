import os
import time
import cv2
import numpy as np
import logging

import math
import logging as log

from face_detection import FaceDetectionModel
from facial_landmarks_detection import FacialLandmarksDetectionModel
from gaze_estimation import GazeEstimationModel
from head_pose_estimation import HeadPoseEstimationModel
#from mouse_controller import MouseController
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
    parser.add_argument("-flag", "--previewFlags", required=False, nargs='+',
                        default=[],
                        help="Specify the flags from fd hp fl ge, like --flag fd hp fl ge (Seperate each flag by space)"
                             "for see the visualization of different model outputs of each frame,"
                             "fd for Face Detection Model, hp for Head Pose Estimation Model"
                             "fl for Facial Landmark Detection Model, ge for Gaze Estimation Model.")
    return parser


def infer_on_stream(args):
    
    try:
    
        # Logger
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
            handlers=[
                logging.FileHandler("time-stats.log"),
                logging.StreamHandler()
            ])

        input_file_path = args.input # Path to image or video file
        logger = log.getLogger()
        previewFlags = args.previewFlags

        input_feeder = input_feeder_func(input_file_path)

        face_detection_instant, head_pose_estimation_instant, facial_landmarks_instant, gaze_estimation_instant = \
        model_instants(args)

        ### Loading and logging the model load time ###
        logging.info("========== Models Load time ==========") 
        start_time = time.time()
        face_detection_instant.load_model()
        logging.info("Face Detection Model: {:.1f}ms".format(1000 * (time.time() - start_time)) )

        start_time = time.time()
        head_pose_estimation_instant.load_model()
        logging.info("Headpose Estimation Model: {:.1f}ms".format(1000 * (time.time() - start_time)) )

        start_time = time.time()
        facial_landmarks_instant.load_model()
        logging.info("Facial Landmarks Detection Model: {:.1f}ms".format(1000 * (time.time() - start_time)) )

        start_time = time.time()
        gaze_estimation_instant.load_model()
        logging.info("Gaze Estimation Model: {:.1f}ms".format(1000 * (time.time() - start_time)) )
        logging.info("========== End ==========") 

        # Open video capture
        input_feeder.load_data()


        # init scene variables
        frame_count = 0

        # Initialize inference time
        face_detection_infer_time = 0
        facial_landmarks_infer_time = 0
        head_pose_estimation_infer_time = 0
        gaze_estimation_infer_time = 0

    #    counter = 0
    #    start_inf_time = time.time()
        #logger.error("Start inferencing on input video.. ")
        
        ### Loop until stream is over ###
        for flag, frame in input_feeder.next_batch():
            if not flag:
                break
            pressed_key = cv2.waitKey(60)

            frame_count += 1
    #        counter = counter + 1

            start_time = time.time()
            face_coordinates, face_image = face_detection_instant.predict(frame.copy())
            face_detection_infer_time += time.time() - start_time

    #         if face_coordinates == 0:
    #             continue

            if type(face_image)==int:
                logger.error("Unable to detect the face.")
                if key==27:
                    break
                continue

            start_time = time.time()
            head_pose_output = head_pose_estimation_instant.predict(face_image)
            head_pose_estimation_infer_time += time.time() - start_time

            start_time = time.time()
            left_eye, right_eye, eye_coords = facial_landmarks_instant.predict(face_image)
            facial_landmarks_infer_time += time.time() - start_time

            start_time = time.time()
            new_mouse_coords, gaze_vector = gaze_estimation_instant.predict(left_eye, right_eye,
                                                                                 head_pose_output)
            gaze_estimation_infer_time += time.time() - start_time

            if len(previewFlags) != 0:
                preview_window = frame.copy()
                if 'fd' in previewFlags:
                    if len(previewFlags) != 1:
                        preview_window = face_image
                    else:
                        cv2.rectangle(preview_window, (face_coordinates[0], face_coordinates[1]),
                                      (face_coordinates[2], face_coordinates[3]), (0, 150, 0), 3)
                if 'fl' in previewFlags:
                    if not 'fd' in previewFlags:
                        preview_window = face_image.copy()
                    cv2.rectangle(preview_window, (eye_coords[0][0], eye_coords[0][1]), (eye_coords[0][2], eye_coords[0][3]),
                                  (150, 0, 150))
                    cv2.rectangle(preview_window, (eye_coords[1][0], eye_coords[1][1]), (eye_coords[1][2], eye_coords[1][3]),
                                  (150, 0, 150))
                if 'hp' in previewFlags:
                    cv2.putText(preview_window,
                                "yaw:{:.1f} | pitch:{:.1f} | roll:{:.1f}".format(head_pose_output[0],
                                                                                 head_pose_output[1],
                                                                                 head_pose_output[2]),
                                (20, 20), cv2.FONT_HERSHEY_COMPLEX, 0.35, (0, 0, 0), 1)
                if 'ge' in previewFlags:

                    yaw = head_pose_output[0]
                    pitch = head_pose_output[1]
                    roll = head_pose_output[2]
                    focal_length = 950.0
                    scale = 50
                    center_of_face = (face_image.shape[1] / 2, face_image.shape[0] / 2, 0)
                    if 'fd' in previewFlags or 'fl' in previewFlags:
                        draw_axes(preview_window, center_of_face, yaw, pitch, roll, scale, focal_length)
                    else:
                        draw_axes(frame, center_of_face, yaw, pitch, roll, scale, focal_length)

            if len(previewFlags) != 0:
                img_hor = np.hstack((cv2.resize(frame, (500, 500)), cv2.resize(preview_window, (500, 500))))
            else:
                img_hor = cv2.resize(frame, (500, 500))
                
            
            # Saving the image 
            cv2.imwrite('output_images/demo.png', img_hor)
            
            
            #cv2.imshow('Visualization', img_hor)
            #mouse_controller_object.move(new_mouse_coords[0], new_mouse_coords[1])

            if pressed_key == 27:
                logger.error("exit key pressed..")
                break
#        inference_time = round(time.time() - start_inf_time, 1)
    #    fps = int(counter) / inference_time
    #     logger.error("counter {} seconds".format(counter))
    #     logger.error("total inference time {} seconds".format(inference_time))
    #     logger.error("fps {} frame/second".format(fps))
    #     logger.error("Video has ended")
#         with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'stats.txt'), 'w') as f:
#             f.write(str(inference_time) + '\n')
#             f.write(str(fps) + '\n')
#             f.write(str(load_total_time) + '\n')

        #logging inference times
        if(frame_count > 0):
            logging.info("========== Models Inference time ==========") 
            logging.info("Face Detection Model:{:.1f}ms".format(1000* face_detection_infer_time/frame_count))
            logging.info("Facial Landmarks Detection Model:{:.1f}ms".format(1000* facial_landmarks_infer_time/frame_count))
            logging.info("Headpose Estimation Model:{:.1f}ms".format(1000* head_pose_estimation_infer_time/frame_count))
            logging.info("Gaze Estimation Model:{:.1f}ms".format(1000* gaze_estimation_infer_time/frame_count))
            logging.info("========== End ==========") 

        #logger.error("VideoStream ended...")
        input_feeder.close()
        cv2.destroyAllWindows()
    
    except Exception as e:
        logging.exception("Error running inference:" + str(e))

# NOTE: draw_axes and build_camera_matrix code implementation gotten from 
# https://knowledge.udacity.com/questions/171017    
def draw_axes(frame, center_of_face, yaw, pitch, roll, scale, focal_length):
    yaw *= np.pi / 180.0
    pitch *= np.pi / 180.0
    roll *= np.pi / 180.0
    cx = int(center_of_face[0])
    cy = int(center_of_face[1])
    r_x = np.array([[1, 0, 0],
                    [0, math.cos(pitch), -math.sin(pitch)],
                    [0, math.sin(pitch), math.cos(pitch)]])
    r_y = np.array([[math.cos(yaw), 0, -math.sin(yaw)],
                    [0, 1, 0],
                    [math.sin(yaw), 0, math.cos(yaw)]])
    r_z = np.array([[math.cos(roll), -math.sin(roll), 0],
                    [math.sin(roll), math.cos(roll), 0],
                    [0, 0, 1]])

    r = r_z @ r_y @ r_x
    camera_matrix = build_camera_matrix(center_of_face, focal_length)
    xaxis = np.array(([1 * scale, 0, 0]), dtype='float32').reshape(3, 1)
    yaxis = np.array(([0, -1 * scale, 0]), dtype='float32').reshape(3, 1)
    zaxis = np.array(([0, 0, -1 * scale]), dtype='float32').reshape(3, 1)
    zaxis1 = np.array(([0, 0, 1 * scale]), dtype='float32').reshape(3, 1)
    o = np.array(([0, 0, 0]), dtype='float32').reshape(3, 1)
    o[2] = camera_matrix[0][0]
    xaxis = np.dot(r, xaxis) + o
    yaxis = np.dot(r, yaxis) + o
    zaxis = np.dot(r, zaxis) + o
    zaxis1 = np.dot(r, zaxis1) + o
    xp2 = (xaxis[0] / xaxis[2] * camera_matrix[0][0]) + cx
    yp2 = (xaxis[1] / xaxis[2] * camera_matrix[1][1]) + cy
    p2 = (int(xp2), int(yp2))
    cv2.line(frame, (cx, cy), p2, (0, 0, 255), 2)
    xp2 = (yaxis[0] / yaxis[2] * camera_matrix[0][0]) + cx
    yp2 = (yaxis[1] / yaxis[2] * camera_matrix[1][1]) + cy
    p2 = (int(xp2), int(yp2))
    cv2.line(frame, (cx, cy), p2, (0, 255, 0), 2)
    xp1 = (zaxis1[0] / zaxis1[2] * camera_matrix[0][0]) + cx
    yp1 = (zaxis1[1] / zaxis1[2] * camera_matrix[1][1]) + cy
    p1 = (int(xp1), int(yp1))
    xp2 = (zaxis[0] / zaxis[2] * camera_matrix[0][0]) + cx
    yp2 = (zaxis[1] / zaxis[2] * camera_matrix[1][1]) + cy
    p2 = (int(xp2), int(yp2))
    cv2.line(frame, p1, p2, (255, 0, 0), 2)
    cv2.circle(frame, p2, 3, (255, 0, 0), 2)
    return frame


def build_camera_matrix(center_of_face, focal_length):
    cx = int(center_of_face[0])
    cy = int(center_of_face[1])
    camera_matrix = np.zeros((3, 3), dtype='float32')
    camera_matrix[0][0] = focal_length
    camera_matrix[0][2] = cx
    camera_matrix[1][1] = focal_length
    camera_matrix[1][2] = cy
    camera_matrix[2][2] = 1
    return camera_matrix
    
    
def input_feeder_func(input_file_path):
    # Checks for live feed
    if input_file_path == 'CAM':
        input_feeder = InputFeeder("cam")

    # Checks for video file
    else:
        input_feeder = InputFeeder("video", input_file_path)
        assert os.path.isfile(input_file_path), "Specified input file doesn't exist"
        
    return input_feeder

def model_instants(args):

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
    
    #mouse_controller_instant = MouseController('medium', 'fast')
    
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