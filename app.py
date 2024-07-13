from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLineEdit, QTextEdit, QRadioButton, QLabel
import sys
import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R
import socket
import copy
from iiwa_control_ import *
import os
import re


class CalibrationTool(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.count = 0
        self.number_of_points = 20
        self.is_last_calibration_real_data = False

        static_base2camera_rot = R.from_euler('xyz', np.random.rand(3) * 45 - np.random.rand(3) * 90, degrees=True).as_matrix() # -45 to 45 degrees
        static_base2camera_trans = np.random.rand(3) - np.random.rand(3) * 0.4  # Random translation up to 0.5 meters / -0.4  to 1.0

        static_target2gripper_rot = R.from_euler('xyz', np.random.rand(3) * 45 - np.random.rand(3) * 90, degrees=True).as_matrix()
        static_target2gripper_trans = np.random.rand(3) * 0.1 - np.random.rand(3) * 0.05 # Random translation up to 0.5 meters / -0.05 to 0.1

        self.T_base2camera = create_homogeneous_transformation(static_base2camera_rot, static_base2camera_trans)
        self.T_target2gripper = create_homogeneous_transformation(static_target2gripper_rot, static_target2gripper_trans)

        self.T_base2gripper = []
        self.T_target2camera = []

        self.robot_poses=[]
        self.LoopX_poses=[]
        self.iiwa = iiwa_control()

        self.next_file_number = get_next_filenumber("./data", f"LoopX_pose_")

    def initUI(self):
        # Main layout
        mainLayout = QVBoxLayout(self)
        upLayout = QHBoxLayout(self)

        # Left side
        leftLayout = QVBoxLayout()
        self.leftTextBox = QTextEdit(self, readOnly=True)
        self.leftTextBox.setMinimumSize(450, 350)
        self.leftTextBox.append("Coordinates for Target2Camera:")
        self.inputLeft = QLineEdit(self)
        self.inputLeft.setPlaceholderText("Coordinates (x, y, z, q.x, q.y, q.z, q.w)")
        leftInputLayout = QVBoxLayout()
        leftInputLayout.addWidget(self.inputLeft)
        leftLayout.addWidget(self.leftTextBox)
        leftLayout.addLayout(leftInputLayout)

        # Right side
        rightLayout = QVBoxLayout()
        self.rightTextBox = QTextEdit(self, readOnly=True)
        self.rightTextBox.setMinimumSize(450, 350)
        self.rightTextBox.append("Coordinates for Base2Gripper:")
        self.inputRight = QLineEdit(self)
        self.inputRight.setPlaceholderText("Coordinates (x, y, z, q.x, q.y, q.z, q.w)")
        rightInputLayout = QVBoxLayout()
        rightInputLayout.addWidget(self.inputRight)
        rightLayout.addWidget(self.rightTextBox)
        rightLayout.addLayout(rightInputLayout)

        self.saveBtn = QPushButton("Save", self)
        self.saveBtn.clicked.connect(self.save_input)
        self.deleteBtn = QPushButton("Delete last coordinates", self)
        self.deleteBtn.clicked.connect(self.delete_last_line)
        buttonsLayout = QHBoxLayout()
        buttonsLayout.addWidget(self.saveBtn)
        buttonsLayout.addWidget(self.deleteBtn)

        # Adding both sides to the main layout
        upLayout.addLayout(leftLayout)
        upLayout.addLayout(rightLayout)

        downLayout = QVBoxLayout()

        resultsLayout = QHBoxLayout()
        self.rotErrorTextBox = QTextEdit(self, readOnly=True)
        self.rotErrorTextBox.setMinimumSize(230, 110)
        self.rotErrorTextBox.append("Rotation Error (degrees): \n")
        self.transErrorTextBox = QTextEdit(self, readOnly=True)
        self.transErrorTextBox.setMinimumSize(230, 110)
        self.transErrorTextBox.append("Translation Error (meters): \n")
        resultsLayout.addWidget(self.rotErrorTextBox)
        resultsLayout.addWidget(self.transErrorTextBox)
        
        self.resRotationTextBox = QTextEdit(self, readOnly=True)
        self.resRotationTextBox.append("Rotation Matrix:\n")
        self.resRotationTextBox.setMinimumSize(230, 110)     
        self.resTranslationTextBox = QTextEdit(self, readOnly=True)
        self.resTranslationTextBox.append("Translation Vector:\n")
        self.resTranslationTextBox.setMinimumSize(230, 110)
        resultsLayout.addWidget(self.resRotationTextBox)
        resultsLayout.addWidget(self.resTranslationTextBox)

        # Simulation layout
        downSimulateLayout = QHBoxLayout()
        self.inputNumberOfInputs = QLineEdit(self)
        self.inputNumberOfInputs.setPlaceholderText("Number of points (Default: 20)")
        self.simulateBtn = QPushButton("Simulate", self)
        self.simulateBtn.clicked.connect(self.calibrate_with_simulation)
        self.radio1 = QRadioButton("Large area")
        self.radio2 = QRadioButton("Small area")
        self.radio1.setChecked(True)
        vbox = QHBoxLayout()
        vbox.addWidget(self.radio1)
        vbox.addWidget(self.radio2)
        self.setLayout(vbox)
        downSimulateLayout.addLayout(vbox)
        downSimulateLayout.addWidget(self.inputNumberOfInputs)
        downSimulateLayout.addWidget(self.simulateBtn)

        downLayout.addLayout(downSimulateLayout)

        testLayout = QHBoxLayout()
        self.testResultTextBox = QTextEdit(self, readOnly=True)
        self.testResultTextBox.setFixedSize(450, 250)
        self.bestTestResultTextBox = QTextEdit(self, readOnly=True)
        self.bestTestResultTextBox.setFixedSize(450, 250)
        self.testSetInput = QLineEdit(self)
        self.testSetInput.setPlaceholderText("Test Set")
        self.testBtn = QPushButton("Test", self)
        self.testBtn.clicked.connect(self.test)
        self.addCoordBtn = QPushButton("Add Coord", self)
        self.addCoordBtn.clicked.connect(self.add_coordinate)
        testInputsLayout = QVBoxLayout()
        testInputsLayout.addWidget(self.testSetInput)
        testInputsLayout.addWidget(self.testBtn)
        testInputsLayout.addWidget(self.addCoordBtn)
        testLayout.addWidget(self.testResultTextBox)
        testLayout.addWidget(self.bestTestResultTextBox)
        testLayout.addLayout(testInputsLayout)
        
        mainLayout.addLayout(upLayout)
        mainLayout.addLayout(buttonsLayout)
        mainLayout.addLayout(resultsLayout)
        mainLayout.addLayout(downLayout)
        mainLayout.addLayout(testLayout)

        self.setLayout(mainLayout)
        self.setWindowTitle('Calibration Tool')

    def save_input(self):
        if self.inputLeft.text() == "" or self.inputRight.text() == "":
            self.saveBtn.setText("Save (Please fill both inputs)")
            return
            
        left_input = f"{self.inputLeft.text()}"
        right_input = f"{self.inputRight.text()}"

        # validate inputs if they are valid floats and right amount
        try: 
            left_coordinates = list(map(lambda x: float(x.strip()), left_input.split(",")))
            right_coordinates = list(map(lambda x: float(x.strip()), right_input.split(",")))
            if len(left_coordinates) != 7 or len(right_coordinates) != 7:
                self.saveBtn.setText("Save (Missing or more axis!)")
                return
        except Exception as e:
            self.saveBtn.setText("Save (Non numerical coordinates!)")
            print(e)
            return

        homogenous_base2gripper = create_homogeneous_transformation(quaternion_to_matrix(right_coordinates[3:]), np.array(right_coordinates[:3]))
        self.T_base2gripper.append(homogenous_base2gripper)
        homogenous_target2camera = create_homogeneous_transformation(quaternion_to_matrix(left_coordinates[3:]), np.array(left_coordinates[:3]))
        self.T_target2camera.append(homogenous_target2camera)
        self.count += 1

        if self.count >= 3:
            self.calibrate()
        else:
            self.print_transformation_results(without_calib=True)

        self.inputLeft.clear()
        self.inputRight.clear()
        
        self.saveBtn.setText("Save")

    def delete_last_line(self):
        if self.count == 0:
            return
        
        self.T_base2gripper.pop()
        self.T_target2camera.pop()
        self.count -= 1

        if self.count >= 3:
            if self.is_last_calibration_real_data:
                self.calibrate(real_data=True)
            else:
                self.calibrate()

        self.inputLeft.clear()
        self.inputRight.clear()

    def calibrate(self, testNo=None, real_data=False):
        # OpenCV's calibrateHandEye to find camera2base transformation
        self.is_last_calibration_real_data = real_data
        R_camera2base, t_camera2base = cv2.calibrateHandEye(
            [t[:3,:3] for t in self.T_base2gripper], [t[:3,3:] for t in self.T_base2gripper],
            [t[:3,:3] for t in self.T_target2camera], [t[:3,3:] for t in self.T_target2camera],
            method=cv2.CALIB_HAND_EYE_TSAI # CALIB_HAND_EYE_TSAI | CALIB_HAND_EYE_PARK
        )

        self.rotationMatrix = R_camera2base
        self.translationVector = t_camera2base

        if not real_data:
            # Calculate the errors
            rotation_error, translation_error = self.calculate_errors(R_camera2base, t_camera2base)
            self.errorRotation = rotation_error
            self.errorTranslation = translation_error
            self.print_transformation_results()
        else:
            self.print_transformation_results(clear_all_boxes=False, real_data=True, testNo=testNo)
        
    def calibrate_with_simulation(self, is_large_area=None, clear_all_boxes=True, testNo=None, generate_coordinates=True, amount_of_points=None):
        self.is_last_calibration_real_data = False
        number_of_points = self.number_of_points if self.inputNumberOfInputs.text() == "" else int(self.inputNumberOfInputs.text())
        if self.inputNumberOfInputs.text() == "0": generate_coordinates = True # to regenerate coordinates for test
        is_large_area = self.radio1.isChecked() if is_large_area is None else is_large_area

        if generate_coordinates: 
            self.generate_coordinates(is_large_area)

        # OpenCV's calibrateHandEye to find camera2base transformation
        R_camera2base, t_camera2base = cv2.calibrateHandEye(
            [t[:3,:3] for t in self.T_base2gripper][:amount_of_points], [t[:3,3:] for t in self.T_base2gripper][:amount_of_points],
            [t[:3,:3] for t in self.T_target2camera][:amount_of_points], [t[:3,3:] for t in self.T_target2camera][:amount_of_points],
            method=cv2.CALIB_HAND_EYE_TSAI # CALIB_HAND_EYE_TSAI | CALIB_HAND_EYE_PARK
        )

        self.rotationMatrix = R_camera2base
        self.translationVector = t_camera2base

        # Calculate the errors
        rotation_error, translation_error = self.calculate_errors(R_camera2base, t_camera2base)
        self.errorRotation = rotation_error
        self.errorTranslation = translation_error

        self.print_transformation_results(clear_all_boxes, testNo, amount_of_points=amount_of_points)
        self.count = number_of_points
               
    def calculate_errors(self, result_camera2base_rot, result_camera2base_trans):    
        # Invert the static_base2camera to get camera2base (for comparison)
        T_camera2base = np.linalg.inv(self.T_base2camera)

        # Calculate the errors
        _rotation_error = rotation_error(T_camera2base[:3,:3], result_camera2base_rot)
        _translation_error = translation_error(T_camera2base[:3,3:], result_camera2base_trans)

        return _rotation_error, _translation_error
    
    def print_transformation_results(self, clear_all_boxes=True, testNo=None , real_data=False, amount_of_points=None, without_calib=False):
        if clear_all_boxes:
            self.clear_all_boxes()
        if amount_of_points is None:
            amount_of_points = len(self.T_target2camera)

        self.leftTextBox.append("\nCoordinates for Target2Camera: \n")
        self.rightTextBox.append("\nCoordinates for Base2Gripper: \n")
        if not real_data:
            self.rotErrorTextBox.append("\nRotation Error (degrees): \n")
            self.transErrorTextBox.append("\nTranslation Error (meters): \n")
            self.resRotationTextBox.append("\nRotation Matrix: \n")
            self.resTranslationTextBox.append("\nTranslation Vector: \n")

        if without_calib == False:
            # Print the transformation matrix
            for i in range(3):
                # Format each row of the "rotation" matrix
                formatted_row = "  ".join(f"{float(val):+07.4f}" for val in self.rotationMatrix[i])
                self.resRotationTextBox.append(formatted_row)
                
                # Format each element of the "translation" vector
                formatted_translation = f"{float(self.translationVector[i][0]):+07.4f}"
                self.resTranslationTextBox.append(formatted_translation)
        self.resRotationTextBox.append("\n")
        self.resTranslationTextBox.append("\n")

        # Print the coordinates for camera2target
        for j in range(amount_of_points):
            # get only 4 decimal points
            camera2target_rot_matrix = matrix_to_quaternion([t[:3,:3] for t in self.T_target2camera][j])
            coordinate_trans = ", ".join([f"{float(coord):.4f}" for coord in [t[:3,3:] for t in self.T_target2camera][j]])
            coordinate_rot = ", ".join([f"{float(coord):.4f}" for coord in camera2target_rot_matrix])
            coordinate = coordinate_trans + ", " + coordinate_rot
            self.leftTextBox.append(f"{str(j+1).rjust(2)} : {coordinate}")

        # Print the coordinates for base2gripper
        for j in range(amount_of_points):
            # get only 4 decimal points
            base2gripper_rot_matrix = matrix_to_quaternion([t[:3,:3] for t in self.T_base2gripper][j])
            coordinate_trans = ", ".join([f"{float(coord):.4f}" for coord in [t[:3,3:] for t in self.T_base2gripper][j]])
            coordinate_rot = ", ".join([f"{float(coord):.4f}" for coord in base2gripper_rot_matrix])
            coordinate = coordinate_trans + ", " + coordinate_rot
            self.rightTextBox.append(f"{str(j+1).rjust(2)} : {coordinate}")

        if not real_data and not without_calib:
            # Print the errors   
            self.rotErrorTextBox.append(f"{self.errorRotation:.4f}")
            self.transErrorTextBox.append(f"{self.errorTranslation:.4f}")

    def generate_coordinates(self, is_large_area=True):
        # Generate known camera2target data
        T_target2camera = []
        T_base2gripper = []
        for i in range(self.number_of_points):
            T_camera2target_single = generate_random_pose(is_large_area)
            T_base2target_single = T_camera2target_single @ self.T_base2camera
            T_base2gripper_single = self.T_target2gripper @ T_base2target_single
            T_base2gripper.append(T_base2gripper_single)
            T_target2camera.append(np.linalg.inv(T_camera2target_single))
            
        self.T_target2camera, self.T_base2gripper = T_target2camera, T_base2gripper

    def test(self):
        self.clear_all_boxes()

        if self.testSetInput.text() not in ["", "-1", "0"]:
            transformations = self.read_transformations()
            self.apply_tests(transformations)
            return
        
        is_large_area = True
        num_of_points_arr = [5, 8, 10, 15, 20, 25]
        self.number_of_points = max(num_of_points_arr)
        error_arr = []
        
        if self.testSetInput.text() == "-1":
            is_large_area = False
        elif self.testSetInput.text() == "0":
            self.reset_program(num_of_points=self.number_of_points)
        else:
            self.generate_coordinates()
        
        for i, num_of_points in enumerate(num_of_points_arr):
            self.calibrate_with_simulation(is_large_area=is_large_area, clear_all_boxes=False, testNo=i + 1, generate_coordinates=False, amount_of_points=num_of_points)
            error_arr.append((self.errorRotation, self.errorTranslation))

        # Print the test results
        self.testResultTextBox.append("\nTest Results: \n*************************************************")
        for i, error in enumerate(error_arr):
            self.testResultTextBox.append("++ Test " + str(i+1) + " ++\n")
            self.testResultTextBox.append("Number of points: " + str(num_of_points_arr[i//2] ) + "   |   Is Large Area:  " + str("True" if i % 2 == 0 else "False") + "\n")
            self.testResultTextBox.append("Rotation Error: " + str(error[0]))
            self.testResultTextBox.append("Translation Error: " + str(error[1]))
            self.testResultTextBox.append("*************************************************")

        # Print the best test result
        best_error = min(error_arr, key=lambda x: x[1])
        self.bestTestResultTextBox.append("\nBest Test Result: \n*************************************************")
        self.bestTestResultTextBox.append(f"Rotation Error: {best_error[0]:0.4f} degrees")
        self.bestTestResultTextBox.append(f"Translation Error: {best_error[1]:0.4f} meters")
        self.bestTestResultTextBox.append("*************************************************")
    
    def reset_program(self, num_of_points=20):
        self.number_of_points = num_of_points

        static_base2camera_rot = R.from_euler('xyz', np.random.rand(3) * 45 - np.random.rand(3) * 90, degrees=True).as_matrix()
        static_base2camera_trans = np.random.rand(3) - np.random.rand(3) * 0.4  # Random translation up to 0.5 meters
        static_target2gripper_rot = R.from_euler('xyz', np.random.rand(3) * 45 - np.random.rand(3) * 90, degrees=True).as_matrix()
        static_target2gripper_trans = np.random.rand(3) * 0.1 - np.random.rand(3) * 0.05 # Random translation up to 0.5 meters
        self.T_base2camera = create_homogeneous_transformation(static_base2camera_rot, static_base2camera_trans)
        self.T_target2gripper = create_homogeneous_transformation(static_target2gripper_rot, static_target2gripper_trans)

        self.generate_coordinates()

        self.clear_all_boxes()
    
    def read_transformations(self):
        # returns 4x4 transformation matrices
        # data is taken as target2cam, gripper2base
        input_file_loopx = "./data/LoopX_pose_" + self.testSetInput.text().strip()
        T_target2camera = []
        try:
            with open(input_file_loopx, "r") as f:
                lines = f.readlines()
                transformation_target2camera = []
                for i, line in enumerate(lines):
                    if i % 5 == 4:  # the line with ""===""
                        T_target2camera.append(np.array(transformation_target2camera))
                        transformation_target2camera = []
                    else:
                        coordinates = line.split(" ")
                        coordinates_arr = list(map(lambda x: float(x.strip()), coordinates))
                        transformation_target2camera.append(coordinates_arr)
            
            input_file_robot = "./data/robot_pose_" + self.testSetInput.text().strip()
            T_base2gripper = []
            with open(input_file_robot, "r") as f:
                lines = f.readlines()
                transformation_gripper2base = []
                for i, line in enumerate(lines):
                    if i % 5 == 4:  # the line with ""===""
                        transformation_base2gripper = np.linalg.inv(np.array(transformation_gripper2base))
                        T_base2gripper.append(np.array(transformation_base2gripper))
                        transformation_gripper2base = []
                    else:
                        coordinates = line.split(" ")
                        coordinates_arr = list(map(lambda x: float(x.strip()), coordinates))
                        transformation_gripper2base.append(coordinates_arr)

        except FileNotFoundError:
            print(f"Error: The file '{input_file_robot}' does not exist.")
        except Exception as e:
            print(f"Error: {e}")
            
        return T_target2camera, T_base2gripper # 4x4 matrices
    
    def apply_tests(self, transformations):
        T_target2camera, T_base2gripper = transformations # 4x4 matrices
        data_set_percentages = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0]
        transformations = [None for _ in range(len(data_set_percentages))]

        for i, data_set_percentage in enumerate(data_set_percentages):
            self.T_base2gripper = T_base2gripper[:int(data_set_percentage * len(T_base2gripper))]
            self.T_target2camera = T_target2camera[:int(data_set_percentage * len(T_target2camera))]
            self.count = int(data_set_percentage * len(T_base2gripper))
            self.calibrate(testNo=i+1, real_data=True)

    def add_coordinate(self):  
        if self.testSetInput.text().strip() == "0":
            # reset case
            self.reset_program()
            self.leftTextBox.append("Coordinates for Target2Camera:")
            self.rightTextBox.append("Coordinates for Gripper2Base: (Note: Base2Gripper is used for calibration)")
            self.next_file_number = get_next_filenumber("./data", f"LoopX_pose_")
            return

        save_path_loopx = f'./data/LoopX_pose_{self.next_file_number}'
        save_path_robot = f'./data/robot_pose_{self.next_file_number}'
        robot_pose = copy.deepcopy(self.iiwa.current_pos)
        robot_pose = vector_to_transform(robot_pose)
        self.T_base2gripper.append(np.linalg.inv(robot_pose)) # add base2gripper for calibration inversed from gripper2base

        vec_loopx = self.get_pose_LoopX()[0]
        LoopX_pose = vector_to_transform(vec_loopx)
        self.T_target2camera.append(LoopX_pose)

        append_matrix(LoopX_pose, save_path_loopx)
        append_matrix(robot_pose, save_path_robot)

        self.robot_poses.append(robot_pose)
        self.LoopX_poses.append(LoopX_pose)

        if len(self.T_target2camera) >= 3:
            self.calibrate(testNo=self.next_file_number, real_data=True)
        else:
            self.print_transformation_results(clear_all_boxes=False, real_data=True, without_calib=True)

    def get_pose_LoopX(self):
        HOST = '192.168.199.4'  # Change this to the IP address you want to listen on
        PORT = 55598  # Change this to the port you want to listen on
        
        cli = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        connected = False
        while not connected:
            try:
                cli.connect((HOST, PORT))
                connected = True
            except Exception as e:
                print(f"Error binding to {HOST}:{PORT}: {e}")
        response  = cli.recv(512)
        data_str = response.decode()
        try:
            positions = self.find_optitrack_poses(data_str)
        except Exception as e:
            print(e)
        if not response:
            print('nothing received')
        return positions  

    def find_optitrack_poses(self, string):
        # List to store the indices of all occurrences
        poses = []

        # Start index for searching
        start_index = 0

        # Find all occurrences of "OptiTrackPositions" in the string
        while True:
            index_start_p = string.find("OptiTrackPositions=", start_index)
            if index_start_p == -1:
                break
            index_start_p = index_start_p + len("OptiTrackPositions=")

            if string.find("|",index_start_p) == -1:
                index_end_p = string.find("\x00",index_start_p)
            else:
                index_end_p = string.find("|",index_start_p)
            if index_end_p == -1:
                break
            numbers = string[index_start_p:index_end_p]
            start_index = index_end_p + 1
            position = np.array(numbers.split(','), dtype=float)/100

            index_start_r = string.find("OptiTrackRotations=", start_index)
            if index_start_r == -1:
                break
            index_start_r = index_start_r + len("OptiTrackRotations=")

            if string.find("|",index_start_r) == -1:
                index_end_r = string.find("\x00",index_start_r)
            else:
                index_end_r = string.find("|",index_start_r)
            if index_end_r == -1:
                break
            numbers = string[index_start_r:index_end_r]
            start_index = index_end_r + 1
            quat = np.array(numbers.split(','), dtype=float)
            first_element = quat[0]  # Remove the first element from the list
            quat = quat[1:]
            quat = np.append(quat, first_element)
            
            
            pose = np.concatenate((position, quat))
            poses.append(pose)
        return np.array(poses)                                    

    def clear_all_boxes(self):
        self.leftTextBox.clear()
        self.rightTextBox.clear()
        self.rotErrorTextBox.clear()
        self.transErrorTextBox.clear()
        self.resRotationTextBox.clear()
        self.resTranslationTextBox.clear()
        self.testResultTextBox.clear()
        self.bestTestResultTextBox.clear()

def quaternion_to_matrix(quaternion):
    # from 4D quaternion (x,y,z,w) to 3x3 rotation matrix
    return R.from_quat(quaternion).as_matrix()

def matrix_to_quaternion(matrix):
    return R.from_matrix(matrix).as_quat()

def vector_to_transform(vector):
    x, y, z, qx, qy, qz, qw = vector

    # Normalize quaternion
    quaternion = np.array([qx, qy, qz, qw])

    # Create rotation matrix from quaternion
    rotation_matrix = R.from_quat(quaternion).as_matrix()

    # Create 4x4 transformation matrix
    transform_matrix = np.eye(4)
    transform_matrix[:3, :3] = rotation_matrix
    transform_matrix[:3, 3] = [x, y, z]

    return transform_matrix   

def transform_to_vector(transform):
    x, y, z = transform[:3, 3]
    quaternion = R.from_matrix(transform[:3, :3]).as_quat()
    return [x, y, z, *quaternion]


def append_matrix(matrix, filename):
    with open(filename, 'a') as file:
        file.write('\n'.join([' '.join(map(str, row)) for row in matrix]))
        file.write('\n' + '='*20 + '\n')

def generate_random_pose(is_large_area=True):
    rotation = R.from_euler('xyz', np.random.rand(3) * 45 - np.random.rand(3) * 90, degrees=True).as_matrix() # -45 to 45 degrees
    if is_large_area:
        translation = np.random.rand(3) * 0.1 - np.random.rand(3) * 0.2  # Full range -0.2 to 0.1
    else:
        translation = np.random.rand(3) * 0.05 - np.random.rand(3) * 0.1 # Small area -0.1 to 0.05

    return create_homogeneous_transformation(rotation, translation)

def rotation_error(R1, R2):
    # Calculate the geodesic distance between two rotation matrices
    rotation_diff = R1.T @ R2
    trace = np.trace(rotation_diff)
    angle = np.arccos((trace - 1) / 2)
    return np.degrees(angle)  # Convert to degrees for easier interpretation

def translation_error(t1, t2):
    # Calculate the Euclidean distance between two translation vectors
    return np.linalg.norm(t1 - t2)

def create_homogeneous_transformation(R, t):
    transformation = np.eye(4)
    transformation[:3, :3] = R
    transformation[:3, 3:] = t.reshape(3,1)
    return transformation

def get_next_filenumber(directory, filename):
    pattern = re.compile(rf'{filename}_(\d+)')
    max_number = 0

    for filename in os.listdir(directory):
        match = pattern.match(filename)
        if match:
            number = int(match.group(1))
            if number > max_number:
                max_number = number

    return max_number + 1

if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = CalibrationTool()
    win.show()
    sys.exit(app.exec_())