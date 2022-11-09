# Video comparison between TLD and PLN
Python code to make a video comparison of a Traditioan Lane Detector (TLD) 
and a Deep Learning Lane Detector (PLN).
# Requirements
* **Python**
* **OpenCV**
* **Tensorflow GPU** and CUDA capable GPU.

# Installation
a. Install conda following the instructions [here](https://conda.io/projects/conda/en/latest/user-guide/install/index.html).

b. Create a new conda environment and activate it.
```shell
conda create --name videocomp_env
conda activate videocomp_env
```
c. Install packages with conda.
```shell
conda install tensorflow-gpu==2.5.0
conda install -c conda-forge opencv==4.6.0
conda install -c nvidia cuda-nvcc==11.8.89
```
d. Clone the repository
```shell
git clone https://github.com/Ign-Es/PLN_TLD_Comparison.git
cd PLN_TLD_Comparison
```
e. Download the trained TuSimple model available [here](https://drive.google.com/drive/folders/1z22kVkSPOqmU6L2_xRgHHQglSe0xRq0i?usp=sharing).
You should download the complete "trained_pln_tusimple" folder and move
it to your work directory.

f. Your work directory should look like this:
```
PLN_TLD_Comparison
├── trained_pln_tusimple
│   └── assests
│   └── variables
│   └── keras_metadata.pb
│   └── saved_model.pb
├── videos
├── src
│   └── LaneDetector.py
│   └── tld.py
├── MakeVideoMix.py
└── MixVideoLaneDetection.py
```
# Get Started
To visualize a comparison of the outputs of TLD and PLN from a given video input,
you may add the video to the "videos" folder and modify line 7 to add the path to the 
video. Then run the python code:
```shell
python MixVideoLaneDetection.py
```