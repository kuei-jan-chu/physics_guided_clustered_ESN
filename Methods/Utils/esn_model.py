import sys
from Config.global_conf import global_params
sys.path.insert(0, global_params.global_utils_path)
from global_utils import *
import torch
import torch.nn as nn
# PRINTING
from functools import partial


print = partial(print, flush=True)

class ESNModel(nn.Module):
    def __init__(self, model):
        super(ESNModel, self).__init__()
        self.input_dim = model.input_dim
        self.reservoir_size = model.reservoir_size
        self.radius = model.radius 
        self.sigma_input = model.sigma_input
        self.sparsity = model.sparsity
        self.p_in = model.p_in
        self.display_output = model.display_output
        self.regularization = model.regularization
        self.mode = model.esn_mode
        self.getInputLayerWeights = model.getInputLayerWeights
        self.getReservoirWeights = model.getReservoirWeights
        self.scaler = model.scaler

        self.W_in =  model.W_in
        self.W_h = model.W_h
        self.W_out = model.W_out

        self.initializeESN()

    def initializeESN(self):
        if self.mode in ["train"]:
            self.initializeReservoir()

    def initializeReservoir(self):
        self.W_in = self.getInputLayerWeights(self.reservoir_size, self.input_dim, self.sigma_input, self.p_in)
        self.W_h = self.getReservoirWeights(self.reservoir_size, self.reservoir_size, self.radius, self.sparsity)
    
    def augmentHidden(self, h):
        h_aug = h.clone()
        h_aug[::2] = h_aug[::2] ** 2.0 
        return h_aug

    def train(self, train_input_sequence, dynamics_length):
        # train_input_sequence has shape [sequence_length, dimension]
        # hidden_state has shape [reservoir_size, 1]

        sequence_length, input_dim = train_input_sequence.shape
        tl = sequence_length - dynamics_length
        reservoir_size = self.reservoir_size

        # Washout period
        if self.display_output == True : print("\nTRAINING: Dynamics prerun...")
        # initial reservoir state, using shape [batch_size=1, reservoir_size ]
        hidden_state = torch.zeros((self.reservoir_size, 1))
        for t in range(dynamics_length):
            if self.display_output == True:
                print("TRAINING - Dynamics prerun: T {:}/{:}, {:2.3f}%".format(t+1, dynamics_length, (t+1)/dynamics_length*100), end="\r")
            current_input = train_input_sequence[t].view(-1, 1)
            hidden_state = torch.tanh(self.W_in @ current_input + self.W_h @ hidden_state)

        H = []
        Y = []
        NORMEVERY = 10
        HTH = torch.zeros((reservoir_size, reservoir_size))
        YTH = torch.zeros((input_dim, reservoir_size))
        
        if self.display_output == True : print("\nTRAINING: Teacher forcing...")
        for t in range(tl - 1):
            if self.display_output == True:
                print("TRAINING - Teacher forcing: T {:}/{:}, {:2.3f}%".format(t+1, tl-1, (t+1)/(tl-1)*100), end="\r")
            current_input = train_input_sequence[t + dynamics_length].view(-1, 1)
            hidden_state = torch.tanh(self.W_in @ current_input + self.W_h @ hidden_state)        
            # Augment hidden state and store in H
            h_aug = self.augmentHidden(hidden_state)
            target = train_input_sequence[t + dynamics_length + 1].view(-1, 1)

            H.append(h_aug[:,0])
            Y.append(target[:,0])       

            if t % NORMEVERY == 0:
                H_batch = torch.stack(H)
                Y_batch = torch.stack(Y)

                HTH += H_batch.T @ H_batch
                YTH += Y_batch.T @ H_batch

                H = []
                Y = []

        if len(H) != 0:
            # Final batch accumulation for pinv
            H_batch = torch.stack(H)
            Y_batch = torch.stack(Y)

            HTH += H_batch.T @ H_batch
            YTH += Y_batch.T @ H_batch


        if self.display_output == True : print("\nTEACHER FORCING ENDED.")

        if self.display_output == True: print("\nTRAINING: COMPUTING THE OUTPUT WEIGHTS...")
        I = torch.eye(HTH.shape[1])
        W_out = YTH @ torch.linalg.pinv(HTH + I * self.regularization)  # Using torch.linalg.pinv for GPU compatibility
        self.W_out = W_out
        return 0
    

    def descaleData(self, sequence, data_mean, data_std):
        sequence = sequence*data_std.unsqueeze(0) + data_mean.unsqueeze(0)
        return sequence
    

    def predictSequence(self, input_sequence, dynamics_length, iterative_prediction_length):
        sequence_length, _ = input_sequence.shape

        # PREDICTION LENGTH
        if sequence_length != iterative_prediction_length + dynamics_length: raise ValueError("Error! N != iterative_prediction_length + dynamics_length")

        prediction_warm_up = []
        hidden_state = torch.zeros((self.reservoir_size, 1))   
        if self.display_output == True:
                print("PREDICTION - Dynamics prerun")  
        for t in range(dynamics_length):
            if self.display_output == True:
                print("PREDICTION - Dynamics prerun: T {:}/{:}, {:2.3f}%".format(t+1, dynamics_length, (t+1)/dynamics_length*100), end="\r")
            current_input = input_sequence[t].view(-1, 1)
            hidden_state = torch.tanh(self.W_in @ current_input + self.W_h @ hidden_state) 
            out = self.W_out @ self.augmentHidden(hidden_state)
            prediction_warm_up.append(out)

        target = input_sequence[dynamics_length:]
        prediction = []
        if self.display_output == True:
                print("\nPREDICTION:")  
        for t in range(iterative_prediction_length):
            if self.display_output == True:
                print("PREDICTION: T {:}/{:}, {:2.3f}%".format((t+1), iterative_prediction_length, (t+1)/iterative_prediction_length*100), end="\r")
            out = self.W_out @ self.augmentHidden(hidden_state)
            prediction.append(out)
            current_input = out
            hidden_state= torch.tanh(self.W_in @ current_input + self.W_h @ hidden_state) 

        prediction_warm_up = torch.stack(prediction_warm_up)[:,:,0]
        prediction = torch.stack(prediction)[:,:,0]

        target_augment = input_sequence
        prediction_augment= torch.cat((prediction_warm_up, prediction), dim=0)

        d_step = calculateGeometricDistance(target, prediction)

        return prediction, target, prediction_augment, target_augment, d_step
    