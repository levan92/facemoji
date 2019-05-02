# facemoji
simple python program to anonymise faces with emojis 

[Demo output video hosted on Gdrive](https://drive.google.com/open?id=1ZpZZtuUPsrvspxh-I9GvbtTPQ-m1Aplr)

## Usage
Quick start:
`python3 run.py -v 0 -e emojis/ `

To see help for arguments:
`python3 run.py -h `

## My Stack
- CUDA 9.0
- cudnn 7.4.2
- cv2 v3.4.0
- tensorflow-gpu==1.13.1
- keras==2.1.3
- dlib==19.9.99 

## Notes on FR model
- Under faceReg directory, openface_nn4_small2.py contains the model architecture in Keras.
- openface_convert_weights.py loads the architecture in Keras, then reads the weights in CSV format (in openface_nn4_small2_csv_weights directory), then loads it into the model. In the script we can then separate save out the weights in a HDF5 (.h5) file. 
- saving the weights separately from the architecture (as compared to saving the entire weight-loaded model) allows portability between Python3 versions. 

## Acknowledgments
- FR network: nn4_small2 from CMU's [Openface](https://cmusatyalab.github.io/openface/) 
- Keras implementation of Openface from [iwantooxxoox](https://github.com/iwantooxxoox/Keras-OpenFace) (love the name)
- [dlib](http://dlib.net/)
- [Deep SORT](https://github.com/nwojke/deep_sort)
