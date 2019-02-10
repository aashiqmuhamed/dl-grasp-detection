# dl-grasp-detection
Implementaion of deep learning for detecting robot grasping

<p align="center"><img width="80%" src="docs/brains-in-a-vat.gif" /></p>

A Python implementation of Deep Learning for Detecting Robot Grasps by Saxena et. al.

### To run training:

1. Update the path to the Cornell Grasping dataset in run.py, create a folder data to store various weights/update the weight directories. 

    ```Shell
    python run
    ```
### To run detection:

Returns the coordinates of a grasping rectangle (using one large NN) and dispalys the rectangle on the image.

```Shell
onePassDectionForInstDefaultParamsDisplay(100,'E:/Data for grasping/data/all','E:/Data for grasping/backgrounds')
 ```
 
None of .m files are being used, were introduced for debugging. 

Still updating onePassDectionForInstDefaultParams, which is the two NN version with no display.
