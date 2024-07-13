# LoopX-Robot Calibration Project

This project performs eye to hand calibration for vision-guided robots, using PyQt for a graphical user interface and OpenCV for the calibration process. The accuracy of the calibration is measured using relative and mean absolute translation errors.

**NOTE:** This project works only for Linux systems. However, simulation part can be run on any opearating system if lines 8 and 35 about iiwa_control are commented out.

## Features

- Select calibration area (full or small) using a graphical interface.
- Perform hand-eye calibration using OpenCV.
- Measure calibration accuracy using relative and mean absolute translation errors.

## Requirements

- Python 3.9 or higher
- PyQt5
- OpenCV
- NumPy
- SciPy
- RosPy (only possible for Linux)

## Installation

For the installation, usage of conda is highly recommended. If you don't have conda installed, you can download it from [here](https://docs.conda.io/en/latest/miniconda.html).

The following steps will guide you through the installation process after installing conda:

1. Clone the repository:
   ```bash
   git clone <repository_url>
   cd <repository_directory
   ```

2. Install the required packages:
   ```bash
    conda create -n loopx-calibration python=3.9
    conda activate loopx-calibration
    pip install -r requirements.txt
   ```

3. Install rospy (only for Linux):
   ```bash
    conda install -c conda-forge ros-rospy
   ```

4. Run the following command to run the project:
   ```bash
   python app.py
   ```

## Usage
- The **"Simulate**" button can be used to simulate a calibration with requested number (default 20) of random coordinates in the specified range.

- The **"Save**" button can be used to add a new pair of coordinates to the calibration function (min 3 points needed for calibration).

- The **"Delete last coordinates**" button can be used to delete the last pair of coordinates and recalculates the calibration if number of points left is greater than or equal to 3.

- The **"Large Area**" and **"Small Area**" radio buttons can be used to select the calibration area and visually compare the results to see which ones performs better in general.

- The **"Test**" button has different use cases. 
    - It can be used to test the calibration with the selected test set. Currently there are 6 datasets collected for this project. Therefore, one of numbers among 1,2,3,4,5,6 can be entered for the **"Test set**" field to test the calibration. 
    - If no value is entered in the **"Test set**" field, the program will just apply a test suite for the calibration. The a set of 25 random coordinates will be generated and the calibration will be tested with 8 iterative subset of these coordinates. The number of coordinates used for each test set is as follows --> [5, 8, 10, 15, 20, 25]
    - If **"-1**" is entered in this field, the program applies simulation for small area as normally test suite is applied with coordinates generated in a large area.
    - If **"0**" is entered in this field, the program resets itself with new random static transformation matrix and recalculates the calibration based on the new transformation matrices.

- The **"Add coord**" button can be used to add a new pair of coordinates collected from Loopx to the calibration function (you are supposed to connect to Loopx beforehand at the moment). The program will automatically add the new pair of coordinates to the calibration function and recalculates the calibration if there are at least 3 coordinates added so far. If **"0**" is entered in **"Test set**" field and clicked on **"Add coord**" button, the program resets itself to the initial state so users will not need to re-run the app again to reset it. 