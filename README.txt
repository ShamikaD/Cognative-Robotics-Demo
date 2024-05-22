Demo: https://www.youtube.com/shorts/oQvSuy0yqSQ 


To run the project:

Pip install mediaPipe (need python 3.7)
Pip install PyTorch
Pip install numpy
Pip install cv2

genfsm "gesture.gsm"
Run gesture in simple_cli


To retrain weights (should not be necessary): 
Unzip the "hands" folder
Make sure the hands folder is in the same directory as train.py
run the python file "train.py"


Citations:

Mediapipe: https://google.github.io/mediapipe/solutions/hands.html
PyTorch: https://pytorch.org/get-started/locally/
numpy: https://numpy.org/install/
OpenCV: https://pypi.org/project/opencv-python/