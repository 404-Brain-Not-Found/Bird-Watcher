# Bird-Watcher
An application that use object detection to identiy birds.

# How to install x86
1) Install python3
2) Clone the repo
3) run `pip install -r requirements.txt`
4) To run the code enter the falling `python main.py {capture device} {target directory}`
  replace {capture device} with the address of the video or camera you want to watch for birds.
  replace {target directory} with the location you want to have the origianl image, image bound box, annotations, and a csv file.

# How to install on PI
1) Install python3
2) Clone the repo
3) Run `pip install numpy==1.19.4 Pillow==8.0.1 opencv-contrib-python==4.4.0.46`
4) Follow the Tensor Flow [guide](https://www.tensorflow.org/lite/guide/python) to install tensorflow-lite
5) To run the code enter the falling `python main.py {capture device} {target directory}`
  replace {capture device} with the address of the video or camera you want to watch for birds.
  replace {target directory} with the location you want to have the origianl image, image bound box, annotations, and a csv file.
