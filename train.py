import numpy as np

import torch
import torch.nn as nn #for neural network

import cv2 #for image manipulation
import mediapipe as mp #for making images into hand parameters
import os #for walking the file tree to read input images
import shutil #for deleting minst file that is auto-created but messes up file walking

out_dim = 10 #possible amount of things that can be predicted
batchSize = 982 #number of images in each batch -> batch mode: this is the entire dataset. Gives longer iteration time but because I'm only training once, I don't care
epochs = 2500 #number of epochs run approximately the amount needed to converge (found through testing)
in_dim = 42 #number of input neurons: x and y for every joint in your hand
hidden_size = 28 #This size was chosen because it was between the input and the output size and follows the convention of 2/3 of the input layer

if os.path.isdir("hands/mnist_data"):
  shutil.rmtree("hands/mnist_data")

#This is a sequential feed forward fully connected neural network with 1 linear layer and 2 hidden layers (added for better accuracy)
#input is a 1d array of 21 neurons, the x and y of every joint on the hand after normalization
class Network(nn.Module):
  def __init__(self, out_dim, in_dim):
    super(Network, self).__init__()
    self.network1 = nn.Sequential(
      nn.Linear(in_dim, hidden_size),
      nn.ReLU(), #relu gets rid of the vanishing gradient issue that is caused by SGD because its derivative is always more than or equal to 0
      nn.Linear(hidden_size, hidden_size),
      nn.Linear(hidden_size, hidden_size),
      nn.ReLU(),
    )
    self.network2 = nn.Sequential(nn.Linear(hidden_size, out_dim),)

  def forward(self, x):
    out = self.network1(x)
    out = out.view(out.size(0), -1)
    out = self.network2(out)
    return out

# Uncomment just one of the two lines below:
#device = torch.device(0)        # GPU board
device = torch.device('cpu')    # regular CPU -> using this for now because I'm testing code on my personal computer
print('device is', device)

model = Network(out_dim, in_dim).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1) #learning rate is 0.1 becuase 0.01 took too long to converge

def getImages(folder = "hands"):
    images = {}
    for thing in os.walk(folder):
        folder = str(thing[0])
        directs = thing[1]
        for direct in directs:
            if direct == ".DS_Store": #fixes mac issue of adding this weird file in every folder
                continue
            for files in os.walk(folder+ "/" +direct):
                for file in files[2]:
                    path = folder + "/" + direct + "/" + file
                    images[path] = (cv2.imread(path), direct)
    return images

# Run MediaPipe Hands.
def getDistances(images):
    tempImgs = []
    tempLabs = []
    mp_hands = mp.solutions.hands
    with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.7) as hands:
        for name, imageTuple in images.items():
            # Convert the BGR image to RGB, flip the image around y-axis for correct 
            # handedness output and process it with MediaPipe Hands.
            if imageTuple[0] is None:
                continue
            image, direct = imageTuple
            
            results = hands.process(cv2.flip(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), 1))
            if not results.multi_hand_landmarks:
                continue
            else:
                # Draw hand landmarks of each hand.
                for hand_landmarks in results.multi_hand_landmarks:
                    temp = []
                    for i in range(21):
                      temp.append(hand_landmarks.landmark[i].x)
                      temp.append(hand_landmarks.landmark[i].y)
                  
                    num = 0
                    if direct == "ahead": num = 0 #flat hand pointing up
                    elif direct == "behind": num = 1 #flat hand pointing down
                    elif direct == "left": num = 2 #flat hand pointing left 
                    elif direct == "right": num = 3 #flat hand pointing right
                    elif direct == "turn": num = 4 #universal okay sign
                    elif direct == "grab": num = 5 #claw hand pointing down
                    elif direct == "happy": num = 6 #thumbs up
                    elif direct == "hello": num = 7 #universal phone hand gesture
                    elif direct == "glow": num = 8 #peace sign
                    elif direct == "mad": num = 9 #three fingers up

                    tempImgs.append(np.array(temp))
                    tempLabs.append(num)
    return((torch.tensor(tempImgs), torch.tensor(tempLabs)))

trainloader = getDistances(getImages())

for epoch in range(epochs):
  runningLoss = 0.0
  correct = 0.0 
  images = trainloader[0]
  labels = trainloader[1]
  images = images.to(device)
  labels = labels.to(device)
  optimizer.zero_grad()
  temp = torch.tensor(np.resize(images,(batchSize,1,in_dim)))
  outputs = model(temp.float())
  loss = criterion(outputs, labels)
  loss.backward()
  optimizer.step()
  runningLoss += loss.item() * labels.shape[0]

  # Calculate success rate for this batch
  outputs = outputs.detach()
  for i in range(batchSize):
    predict = torch.argmax(outputs[i,:])
    if predict == labels[i]:
      correct += 1
  print('{:2}  loss = {:8.2f}  correct = {:.2f} %'.format(
        epoch, runningLoss, (correct*100)/batchSize))

filename = 'weights.pt'
torch.save(model.to('cpu').state_dict(), filename)
print('Wrote',filename)