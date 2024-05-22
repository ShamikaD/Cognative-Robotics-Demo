import time
import numpy as np
import math
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import mediapipe as mp #for making images into hand parameters
import cv2


from cozmo_fsm import *
    
class gesture(StateMachineProgram):
    def __init__(self):
        super().__init__()
        self.robot.camera.color_image_enabled = True

    def user_image(self,image,gray):
        self.robot.myimage = image #gets the colour image of your hand from cozmo's camera

    class ShowImage(StateNode):
        def start(self,event):
            super().start(event)
            print("looking for hand...")
            boolFlag = True #to ensure that none of the functions run if a hand is not detected

            #model definition
            class Network(nn.Module):
              def __init__(self, out_dim, in_dim, hidden_size):
                super(Network, self).__init__()
                self.network1 = nn.Sequential(
                  nn.Linear(in_dim, hidden_size),
                  nn.ReLU(),
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

            #model creation (dimentions match those in the training python file so that the pretrainedweights match)
            device = torch.device(0)
            in_dim = 42
            out_dim = 10
            hidden_size = 28

            model = Network(out_dim, in_dim, hidden_size).to(device)

            #load the weights
            model.load_state_dict(torch.load("weights.pt"))
            model.eval() 
            

            #image processing
            image = self.robot.myimage
            mp_hands = mp.solutions.hands
            with mp_hands.Hands(
                static_image_mode=True,
                max_num_hands=1,
                min_detection_confidence=0.3) as hands:
                cv2.imwrite("img.png", (cv2.flip(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), 1)))
                results = hands.process(cv2.flip(cv2.cvtColor(cv2.imread("img.png"), cv2.COLOR_BGR2RGB), 1))
                # Draw hand landmarks.
           
                if results.multi_hand_landmarks is not None:
                    boolFlag = False
                    hand_landmarks = results.multi_hand_landmarks[0] #there should only be one element in the array but we want to get rid of the outside array
                    image_height, image_width, image_depth = image.shape
                    
                    temp = []
                    for i in range(21): #21 is the number of points on the hand that mediapipe detects
                        temp.append(hand_landmarks.landmark[i].x)
                        temp.append(hand_landmarks.landmark[i].y)
                        #not using z coordinates because cozmo's grainy camera makes it hard for mediapipe to "see" depth because of the messed up colouring of shadows and such
                    
                    ''' 
                    #uncomment to see the graphical representation of the hand image that Cozmo sees!
                    annotated_image = image.copy()
                    mp_drawing = mp.solutions.drawing_utils
                    mp_drawing_styles = mp.solutions.drawing_styles
                    mp_drawing.draw_landmarks(
                      annotated_image,
                      hand_landmarks,
                      mp_hands.HAND_CONNECTIONS,
                      mp_drawing_styles.get_default_hand_landmarks_style(),
                      mp_drawing_styles.get_default_hand_connections_style())
                    cv2.imwrite(
                        '/tmp/annotated_image' + '.png', cv2.flip(annotated_image, 1))
                        
                    for hand_world_landmarks in results.multi_hand_world_landmarks:
                      mp_drawing.plot_landmarks(
                        hand_world_landmarks, mp_hands.HAND_CONNECTIONS, azimuth=5)
                    '''

            if not boolFlag:
                vect = np.array(temp)
                
                #predicting number based on image
                temp = torch.tensor(np.resize(vect,(1,1,in_dim)))
                temp = temp.to(device)
                output = model(temp.float())
                output = output.detach()
                predict = torch.argmax(output)
                print(predict)
                print(predict.item())
                self.post_data(predict)
            else:
                self.post_failure()
            
            
    class Action(Turn):
        def start(self, event=None):
            if isinstance(event, DataEvent):
                if event.data == 0: #forward
                    self.angle = degrees(0)
                    super().start(event)
                elif event.data == 3: #right
                    self.angle = degrees(90)
                    super().start(event)
                elif event.data == 2: #left
                    self.angle = degrees(-90)
                    super().start(event) 
                elif event.data == 1: #back
                    super().start(event)
                    self.post_success()
                elif event.data == 4: #spin
                    super().start(event)
                    self.post_failure()
                else:
                    super().start(event)
                    self.post_data(event.data)
                    
    class Speak(Say):
        def start(self, event=None):
            if isinstance(event, DataEvent):
                if event.data == 5: #sad
                    robot.play_anim_trigger(CodeLabDejected)
                    self.text = 'Let me find a cube'
                    super().start(event)
                elif event.data == 6: #happy
                    robot.play_anim_trigger(cozmo.anim.Triggers.CodeLabHappy)
                    self.text = 'Thanks for the thumbs up'
                    super().start(event)
                elif event.data == 7: #hello
                    self.text = 'hello? Who\'s on the phone'
                    super().start(event)
                elif event.data == 8: #peace
                    self.text = 'peace'
                    super().start(event)
                else: #mads
                    robot.play_anim_trigger(cozmo.anim.Triggers.CodeLabCelebrate)
                    self.text = 'WooHoo'
                    super().start(event)
                    

    def setup(self):
        #         loop: StateNode()
        #         loop =T(1)=> SetHeadAngle(20) =C=> SetLiftHeight(0) =C=> img =D=> act
        #         img =F=> loop
        #         img: self.ShowImage()
        #         
        #         act: self.Action()
        #         act =C=> Forward(100) =C=> loop
        #         act =S=> Forward(-100) =C=> loop
        #         act =F=> Turn(360) =C=> loop
        #         
        #         act =D=> self.Speak() =C=> loop
        # 
        
        # Code generated by genfsm on Thu Apr 28 21:00:02 2022:
        
        loop = StateNode() .set_name("loop") .set_parent(self)
        setheadangle1 = SetHeadAngle(20) .set_name("setheadangle1") .set_parent(self)
        setliftheight1 = SetLiftHeight(0) .set_name("setliftheight1") .set_parent(self)
        img = self.ShowImage() .set_name("img") .set_parent(self)
        act = self.Action() .set_name("act") .set_parent(self)
        forward1 = Forward(100) .set_name("forward1") .set_parent(self)
        forward2 = Forward(-100) .set_name("forward2") .set_parent(self)
        turn1 = Turn(360) .set_name("turn1") .set_parent(self)
        speak1 = self.Speak() .set_name("speak1") .set_parent(self)
        
        timertrans1 = TimerTrans(1) .set_name("timertrans1")
        timertrans1 .add_sources(loop) .add_destinations(setheadangle1)
        
        completiontrans1 = CompletionTrans() .set_name("completiontrans1")
        completiontrans1 .add_sources(setheadangle1) .add_destinations(setliftheight1)
        
        completiontrans2 = CompletionTrans() .set_name("completiontrans2")
        completiontrans2 .add_sources(setliftheight1) .add_destinations(img)
        
        datatrans1 = DataTrans() .set_name("datatrans1")
        datatrans1 .add_sources(img) .add_destinations(act)
        
        failuretrans1 = FailureTrans() .set_name("failuretrans1")
        failuretrans1 .add_sources(img) .add_destinations(loop)
        
        completiontrans3 = CompletionTrans() .set_name("completiontrans3")
        completiontrans3 .add_sources(act) .add_destinations(forward1)
        
        completiontrans4 = CompletionTrans() .set_name("completiontrans4")
        completiontrans4 .add_sources(forward1) .add_destinations(loop)
        
        successtrans1 = SuccessTrans() .set_name("successtrans1")
        successtrans1 .add_sources(act) .add_destinations(forward2)
        
        completiontrans5 = CompletionTrans() .set_name("completiontrans5")
        completiontrans5 .add_sources(forward2) .add_destinations(loop)
        
        failuretrans2 = FailureTrans() .set_name("failuretrans2")
        failuretrans2 .add_sources(act) .add_destinations(turn1)
        
        completiontrans6 = CompletionTrans() .set_name("completiontrans6")
        completiontrans6 .add_sources(turn1) .add_destinations(loop)
        
        datatrans2 = DataTrans() .set_name("datatrans2")
        datatrans2 .add_sources(act) .add_destinations(speak1)
        
        completiontrans7 = CompletionTrans() .set_name("completiontrans7")
        completiontrans7 .add_sources(speak1) .add_destinations(loop)
        
        return self
