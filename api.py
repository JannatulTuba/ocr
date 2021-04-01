from flask import Flask
from flask import render_template
import os
from flask import request


import os
import re
import torch
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
import numpy as np
from tqdm import tqdm
import torch.optim as optim
from torch.utils.data.sampler import SubsetRandomSampler
import json
import pickle
import Levenshtein as Lev
import torch.utils.data as data
import cv2
import numpy as np
import os
import random
from matplotlib import pyplot as plt
import math
import torch.nn.init as init
import torch.nn as nn
import torch.nn.functional as F

app = Flask (__name__)
UPLOAD_FOLDER ="static"
DEVICE = "cuda"
MODEL = None



#########################################################
with open('inv_grapheme_dict.pkl', 'rb') as handle:
    inv_grapheme_dict = pickle.load(handle)
    print(inv_grapheme_dict)


grapheme_dict = {v: k for k, v in inv_grapheme_dict.items()}


#########DO NOT LOOK At These Functions######
############################################
##############################################
#WER functions
def compute_wer(predictions, labels):
    total_dist = 0
    dist_1 = 0
    dist_2 = 0
    dist_3 = 0
    dist_0 = 0
    dist_4 = 0
    dist_5 = 0
    dist_6 = 0


    
    #Check if prediction and original label are same
    assert len(predictions) == len(labels)

    for i in range(len(predictions)):
        edit_dist = Lev.distance(predictions[i], labels[i])
        #print(edit_dist)
        #print(predictions[i])
        #print(labels[i])

        total_dist += edit_dist

        #print(type(edit_dist))
        if (edit_dist == 1):
            dist_1 += 1
        elif (edit_dist == 0):
            dist_0 += 1
        elif (edit_dist == 2):
            dist_2 += 1
        elif (edit_dist == 3):
            dist_3 += 1
        elif (edit_dist == 4):
            dist_4 += 1
        elif (edit_dist == 5):
            dist_5 += 1
        elif (edit_dist == 6):
            dist_6 += 1

    word_error_rate = ( total_dist/len(predictions) )

    return word_error_rate, dist_0, dist_1, dist_2, dist_3,  dist_4, dist_5, dist_6



#Absolute matching function
def absolute_word_match(predictions, labels):
    count_correct = 0
    for x, y in zip(predictions, labels):
        #print(x)
        #print(y)
        if(x==y):
            count_correct += 1
    print("Absolute word match count is {}".format(count_correct) )

    return count_correct


def preprocess_data(data_dir):
    #grapheme_dict = {}
    labels = []
    words = []
    lengths = []
    count = 2
  #  filenames = os.listdir(data_dir)
   # filenames = sorted(filenames, key=lambda x: int(x.split('_')[0]))
    ###################New added####################
    base=os.path.basename(data_dir)
    filenames_withext = os.path.splitext(base)
    name  = os.path.splitext(base)[0]
    ##############################################
#     head, tail = os.path.split(data_dir)
#     filenames = tail
#     print(tail)

  
   # print(name)
    #grapheme_dict['<eow>'] = 1

  
  #  print(name)
    curr_word = name.split('_')[1]
 #   print("current words")
  #  print(curr_word)

     #############################   
    curr_label = []
    words.append(curr_word)
    graphemes = extract_graphemes(curr_word)
        #if 'স্ক্র্' in graphemes:
        #    print(curr_word)
        #    print(graphemes)
    for grapheme in graphemes:
        if grapheme not in grapheme_dict:
                #grapheme_dict[grapheme] = count
                #For inference
                #curr_label.append(count)
            curr_label.append(0)
                #count += 1
        else:
            curr_label.append(grapheme_dict[grapheme])
    lengths.append(len(curr_label))
    labels.append(curr_label)
    #############################

    inv_grapheme_dict = {v: k for k, v in grapheme_dict.items()}
    return grapheme_dict, inv_grapheme_dict, words, labels, lengths


def decode_prediction(preds, inv_grapheme_dict, raw = False):
    #print("Preds:",preds)
    #print("predLen:",len(preds))
    grapheme_list = []
    pred_list = []
    for i in range(len(preds)):
        if preds[i] != 0 and preds[i] != 1 and (not (i > 0 and preds[i - 1] == preds[i])):
            grapheme_list.append(inv_grapheme_dict.get(preds[i]))
            #print("preds[i]",preds[i])
            #print("grapheme_list",grapheme_list)
            pred_list.append(preds[i])
    ##################Cases that hold NOne types####################
    # #print(pred_list)
    # if(len(pred_list) != 0):
    #     return pred_list, ''.join(grapheme_list)
    # else:
    #     return None
    
    return pred_list, ''.join(grapheme_list)



####################################Extracts graphemes(letters) from dataset and throws away useless ones####
###################You can try to understand it but will require Bangla Grammer skills and good logic ######
############################################################################################################


def extract_graphemes(word):
    support_chars = ['্', 'ং', 'ঃ', 'ঁ', 'ি', 'ু', 'ূ', 'ৃ', 'ে', 'ো', 'ৌ' ,'ী', 'া', 'ে', 'ৈ']
    ref_chars = [ '্য', '্র', 'র্', 'য', 'র']
    unicode_garbage = ['\x02', '\x03', '\x06', '\x08', '\x10', '\x12', '&', '¡',
                        '¤', '¥', '¦', '©', '¬', '\xad', '®', '¯', 'Ä', 'Í', 'ä', 'æ', 'è', 'ø', 'ÿ',
                        'œ', 'š', 'Ÿ', 'ƒ', 'β', '॥', '\u09e4', '\u200b', '\u200d', '\u200f', '\uf020',
                        '\uf02d', '�', '\u200b', '\u200c', '\u09e5']
    
    chars = []
    i = 0
    prev_ref = False

    while(i < len(word)):
        if word[i] != support_chars[0] and word[i] not in unicode_garbage:
            if i+1 < len(word):
                if word[i+1] != support_chars[0]:
                    if word[i+1] == ref_chars[-1] and i+2 < len(word):
                        if word[i+2] == support_chars[0]:
                            chars.append(word[i])
                            chars.append(ref_chars[2])
                            i += 2
                            prev_ref = True
                        else:
                            chars.append(word[i])
                            i += 1
                    else:
                        chars.append(word[i])
                        i += 1
                elif word[i+1] == support_chars[0] and word[i] not in support_chars[0:]:
                    
                    previous = False
                    isSupport = True
                    idx = i+1
                    if idx<len(word):
                        while(isSupport):
                            if idx<len(word):
                                #print(word[i], word[idx], i, idx)
                                if (word[idx] == support_chars[0] or word[idx] == ref_chars[4]) and idx+1 < len(word):
                                    if word[idx] == support_chars[0] and word[idx-1] == ref_chars[-1]:
                                        if not previous:
                                            if i != idx:
                                                chars.append(word[i:(idx-1)])
                                            chars.append(ref_chars[2])
                                            idx += 1
                                            i = idx
                                            continue
                                    if word[idx] == ref_chars[-1]:

                                        if word[idx+1] != support_chars[0]:
                                            chars.append(ref_chars[-1])
                                        idx += 1
                                        i = idx
                                        continue
                                    if word[idx+1] == ref_chars[3]:
                                        if i != idx:
                                            chars.append(word[i:idx])
                                        chars.append(ref_chars[0])
                                        idx += 2
                                        i = idx
                                        # print(i)
                                        # print(idx)
                                        continue
                                    if word[idx+1] == ref_chars[4]:
                                        # print(chars)
                                        if i != idx:
                                            chars.append(word[i:idx])
                                        chars.append(ref_chars[1])
                                        idx += 2
                                        i = idx
                                        previous = True
                                        continue
                                    if word[idx+1] == '\u200c':
                                        if i != idx:
                                            chars.append(word[i:idx])
                                        i = idx+2
                                        isSupport = False

                                    idx += 2
                                else:
                                    isSupport= False
                            else:
                                isSupport = False
                    if i != idx:
                        chars.append(word[i:idx])
                    i = idx
                else:
                    if word[i] in support_chars[0:]:
                        chars.append(word[i])
                    i += 2
            else:
                chars.append(word[i])
                i += 1
        else:
            if word[i]== support_chars[0]:
                if prev_ref:
                    prev_ref = False
                    i += 1
                    continue
                chars.append(word[i])
                i+=1
                continue
            else:
                i+=1
                continue
        
    return chars
#######################################################
#########OCR Dataset Class###################################################
#########Helper functions to convert your images#############################
#########to the desired format###############################################

class  OCRDataset(data.Dataset):
    def __init__(self, img_dir):

        self.img_dir = img_dir
      #  print("this is from ocrdataset")
       # print( self.img_dir)
    
        # self.text_dir = text_dir
        self.inp_h = 32
       
        self.inp_w = 128
        self.mean = np.array(0.588, dtype=np.float32)
        self.std = np.array(0.193, dtype=np.float32)
        
        tail = img_dir.split('_')[1]
        tail_list = []
        
        tail_list.append(tail)
     
  #      print(tail_list)
       # head, tail = os.path.split(img_dir)
        self.images =tail_list
        
#         path = os.path.join(head, tail)
#         print("this is path:")
#         print(path)
        #self.images = sorted(os.listdir(img_dir), key=lambda x: int(x.split('_')[0]))
       
    
    def __len__(self):
        return len(self.images)
    

    def __getitem__(self, idx):
        img_name = self.images[idx]
        #print(img_name)

        from skimage import io

        img = io.imread(self.img_dir)



        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


        img_h, img_w = img.shape

        #Resize to input size for network (32,128,1)
        img = cv2.resize(img, (0,0), fx=self.inp_w / img_w, fy=self.inp_h / img_h, interpolation=cv2.INTER_CUBIC)
        img = np.reshape(img, (self.inp_h, self.inp_w, 1))

        #Normalize by mean and standard deviation
        img = img.astype(np.float32)
        img = (img/255. - self.mean) / self.std

        #Reshape to tensor format supported by Pytorch (C, H, W)
        img = img.transpose([2, 0, 1])
        
#         print(img)
#         print(img_name)
#         print(idx)
        return img, img_name, idx

########################################

############Helper function to pad your label lists######################################
############Get all necessary labels from your datasets#################################

def get_padded_labels(idxs, grapheme_dict, inv_grapheme_dict, words, labels, lengths):
    batch_labels = []
    batch_lengths = []
    batch_words = []
    maxlen = 0
    #print("idxs", idxs)
    for idx in idxs:
        batch_labels.append(labels[idx])
        batch_words.append(words[idx])
        #print("word :",words[idx])
        batch_lengths.append(len(labels[idx]))
        maxlen = max(len(labels[idx]), maxlen)
    
    #changed [1]*(maxlen-len(batch_labels[i])) to [0]*(maxlen-len(batch_labels[i]))
    #Alls good
    for i in range(len(batch_labels)):
        batch_labels[i] = batch_labels[i] + [1]*(maxlen-len(batch_labels[i]))

    return batch_words, batch_labels, batch_lengths, inv_grapheme_dict
#################################################################################    
#######################Some module that predicts sequences from the compacted feature rich version of ########
#######################image################################################################################


class BidirectionalLSTM(nn.Module):
    # Inputs hidden units Out
    def __init__(self, nIn, nHidden, nOut):
        super(BidirectionalLSTM, self).__init__()

        self.rnn = nn.LSTM(nIn, nHidden, bidirectional=True)
        self.embedding = nn.Linear(nHidden * 2, nOut)

    def forward(self, input):
        recurrent, _ = self.rnn(input)
        T, b, h = recurrent.size()
        t_rec = recurrent.view(T * b, h)

        output = self.embedding(t_rec)  # [T * b, nOut]
        output = output.view(T, b, -1)  ##marked
        # print(output.shape)
        return output


#######################Forward part is the real architecture. ################################
#######################The functions before that are used to #################################
#######################declare the feature extracting CNN    #################################

class CRNN(nn.Module):
    def __init__(self, imgH, nc, nclass, nh, n_rnn=2, leakyRelu=False):
        super(CRNN, self).__init__()
        assert imgH % 16 == 0, 'imgH has to be a multiple of 16'
        
        
        #########################Convolutional Backbone Declaration############################
        
        
        ###kernel value for every layer
        ks = [3, 3, 3, 3, 3, 3, 2]
        
        ###padding value for every layer
        ps = [1, 1, 1, 1, 1, 1, 0]
        
        ###stride value for every layer
        ss = [1, 1, 1, 1, 1, 1, 1]
        
        ###channel value for every layer
        nm = [64, 128, 256, 256, 512, 512, 512]

        ##Sequential is good way to list layers one after another.
        ##To actually understand the syntax of Pytorch. we would
        ## suggest learning two things: 
        
        ## Syntax of Python classes and objects
        ## https://www.youtube.com/watch?v=wfcWRAxRVBA&list=PLBZBJbE_rGRWeh5mIBhD-hhDwSEDxogDg&index=9
        
        ## For Pytorch, there's the official documents
        ## But this medium article is enough to be honest
        ## https://towardsdatascience.com/pytorch-how-and-when-to-use-module-sequential-modulelist-and-moduledict-7a54597b5f17
        
        
        cnn = nn.Sequential()

        def convRelu(i, batchNormalization=False):
            nIn = nc if i == 0 else nm[i - 1]
            nOut = nm[i]
            cnn.add_module('conv{0}'.format(i),
                           nn.Conv2d(nIn, nOut, ks[i], ss[i], ps[i]))
            if batchNormalization:
                cnn.add_module('batchnorm{0}'.format(i), nn.BatchNorm2d(nOut))
            if leakyRelu:
                cnn.add_module('relu{0}'.format(i),
                               nn.LeakyReLU(0.2, inplace=True))
            else:
                cnn.add_module('relu{0}'.format(i), nn.ReLU(True))

        convRelu(0)
        
        ##Output shape is 512X1X33
        ##(Batch X Height X Width)
        
        
        cnn.add_module('pooling{0}'.format(0), nn.MaxPool2d(2, 2)) 
        convRelu(1)
        cnn.add_module('pooling{0}'.format(1), nn.MaxPool2d(2, 2)) 
        convRelu(2, True)
        convRelu(3)
        cnn.add_module('pooling{0}'.format(2),
                       nn.MaxPool2d((2, 2), (2, 1), (0, 1)))
        convRelu(4, True)
        convRelu(5)
        cnn.add_module('pooling{0}'.format(3),
                       nn.MaxPool2d((2, 2), (2, 1), (0, 1))) 
        convRelu(6, True)  # 512x1x16

        self.cnn = cnn
        
        
        #########################Convolutional Backbone Declaration############################
        
        
        
        #########################Inherit LSTM function from LSTM function######################
        
        self.rnn = nn.Sequential(
            BidirectionalLSTM(512, nh, nh),
            BidirectionalLSTM(nh, nh, nclass))
        
        #########################Inherit LSTM function from LSTM function######################

    def forward(self, input):

        #input is the input image in (Batch, Channel, Height, Width) form
        
        
        #conv = Feature extracted by Convolutional Network
        conv = self.cnn(input)
        
        
        #######convert feature(conv) so LSTM can read it##################
        
        b, c, h, w = conv.size()
        #print(conv.shape)
        #make sure height is 1, we will predicting along the sequnce
        assert h == 1, "the height of conv must be 1"
        
        
        conv = conv.squeeze(2) # b *512 * width
        conv = conv.permute(2, 0, 1)  # [w, b, c]
        
        #############################################################
        
        
        #############Send to LSTM then###############################
        #############softmax it to get ##############################
        #############probability between 0 and 1#####################
        ###Softmax across dimension 2 because it will have###########
        ####288 possible labels and they will have values############
        ####(probability distribution)(between 0 and 1)##############
        output = F.log_softmax(self.rnn(conv), dim=2)

        return output

    
###########weight initialization helps model achieve better###########
#############gradients gradually, experiment with HE intialization####
#############Xavier init etc, if possible#############################
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

 
#def weight_init(m): 
#	if isinstance(m, nn.Linear):
#		size = m.weight.size()
#		fan_out = size[0] # number of rows
#		fan_in = size[1] # number of columns
#		variance = np.sqrt(2.0/(fan_in + fan_out))
#		m.weight.data.normal_(0.0, variance)        
            
        

        
#####Send model after intializing weight##############################        
def get_crnn():
    
    #(Initial Image Height, Feature Height, Labels, LSTM hidden Layer)
    model = CRNN(32, 1, 203, 256)
    model.apply(weights_init)

    return model
###############################################################
import torch
#from utils import preprocess_data, decode_prediction, compute_wer, absolute_word_match
#from model import get_crnn
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
import numpy as np
from tqdm import tqdm
import torch.optim as optim
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn.functional as F
import torch.nn as nn

import json
import pickle


#import dominate
#from dominate.tags import *
import os


#############Reproducability#######################

torch.manual_seed(42)
np.random.seed(42)

###################################################



model = get_crnn()

model = model.cuda()

with open('inv_grapheme_dict.pkl', 'rb') as handle:
    inv_grapheme_dict = pickle.load(handle)
    print(inv_grapheme_dict)


grapheme_dict = {v: k for k, v in inv_grapheme_dict.items()}


#epoch


#Path to model
model.load_state_dict(torch.load('epoch59.pth' )) #, map_location='cpu'
model.eval()

criterion = torch.nn.CTCLoss(blank =0, reduction='mean', zero_infinity = True)
criterion = criterion.cuda()

#ocr_dataset = OCRDataset('dataset/images')
#/home/imr555/Desktop/Apurba_Job/Day_12/model_inference/WordLevelBanglaOCRv1/out_synthetic_seperate
#ocr_dataset = OCRDataset('uniquekalpurush9bin')

#three_paths
# /home/imr555/Desktop/Apurba_Job/Day_12/model_inference/WordLevelBanglaOCRv1/out_synthetic_unseen
# /home/imr555/Desktop/Apurba_Job/Day_12/model_inference/WordLevelBanglaOCRv1/out_unseen_real

def img_path(path):
    ocr_dataset = OCRDataset(path)
    #print(ocr_dataset)
    #print(len(ocr_dataset[0]))
    #print(len(ocr_dataset[1]))

    file_namelist = []


    # for i in range(64):
    #     #print(ocr_dataset[i][1])

    #     file_namelist.append(ocr_dataset[i][1])


    #Test Path Mendeley = C:/Users/monsur/Desktop/croppingimage/rawimgdataset
    #Test Path BengaliAI = C:/Users/monsur/Desktop/BengaliAiHandwrittenDataset/TrainingTesting/testingcon
    #test path mendeley converted = C:/Users/monsur/Desktop/LearningPytorch/HandwritingDataset

    #_, _, words_i, labels_i, lengths_i = preprocess_data('uniquekalpurush9bin')

    _, _, words_i, labels_i, lengths_i = preprocess_data(path)

    #sanity check
    # print("current new word")
    # print(words_i)
    # print("word len")
    # print(len(words_i))
    # print("labels")
    # print(labels_i)


    inference_loader = torch.utils.data.DataLoader(ocr_dataset, batch_size=512)
    #print(type(inference_loader))
    softs = nn.Softmax(dim=2)



    def validate(metrics1, metrics2,file_namelist,train_loss=None):
        epoch=None
        with torch.no_grad():
            y_true = []
            y_pred = []
            pred_ = []
            label_ = []
            total_wer = 0
            distance_list = []
            preds_beam = []

            print("***Epoch: {}***".format(epoch))
            batch_loss = 0
            for i, (inp, img_names, idx) in enumerate(tqdm(inference_loader)):

                inp = inp.cuda()
                batch_size = inp.size(0)
    #             print("batch_size: ")
    #             print(batch_size)
               # print(img_names)
                idxs = idx.detach().numpy()
                img_names = list(img_names)
                words, labels, labels_size, inv_grapheme_dict_ = get_padded_labels(idxs, grapheme_dict, inv_grapheme_dict, words_i, labels_i, lengths_i)
                preds = model(inp)
                labels = torch.tensor(labels, dtype=torch.long)
                labels.cuda()
                labels_size = torch.tensor(labels_size, dtype=torch.long)
                labels_size.cuda()
                preds_size = torch.tensor([preds.size(0)] * batch_size, dtype=torch.long) 
                preds_size.cuda()

                #validation loss
                loss = criterion(preds, labels, preds_size, labels_size)
                #print(loss)
                batch_loss += loss.item()
                #print(loss.item())

                _, preds = preds.max(2)
                preds = preds.transpose(1, 0).contiguous().detach().cpu().numpy()
                labels = labels.detach().numpy()

                for i in range(len(preds)):
                    decoded, _ = decode_prediction(preds[i], inv_grapheme_dict)
                    for x,y in zip(decoded, labels[i]):
                        y_pred.append(x)
                        y_true.append(y)
                    _, decoded_pred_ = decode_prediction(preds[i], inv_grapheme_dict)
                    #print(inv_grapheme_dict)
                    _, decoded_label_ = decode_prediction(labels[i], inv_grapheme_dict)
                    #print(decoded_label_)

                    pred_.append(decoded_pred_)
                    label_.append(decoded_label_)

            valid_loss = batch_loss/batch_size
            print("Epoch Validation loss: ", valid_loss) #batch_size denominator 32
            print("\n")
            #print(pred_)
            #print(label_)
            total_wer, dist_0, dist_1, dist_2, dist_3, dist_4, dist_5, dist_6 = compute_wer(pred_, label_)
            print("Total_Word_Error_Rate: %.4f" % total_wer)

            distance_list.append(dist_0)
            distance_list.append(dist_1)
            distance_list.append(dist_2)
            distance_list.append(dist_3)
            distance_list.append(dist_4)
            distance_list.append(dist_5)
            distance_list.append(dist_6)

            print(distance_list)
            #with open('EditDistanceList123.txt', 'w') as edit:
            #    for x in distance_list:
            #        edit.write(distance_list[x])


            report = classification_report(y_true, y_pred, labels = np.arange(1,370), zero_division=0)
            f1_micro = f1_score(y_true, y_pred, average = 'micro', zero_division=0)
            f1_macro = f1_score(y_true, y_pred, average = 'macro', zero_division=0)
            accuracy = accuracy_score(y_true, y_pred)


            #Absolute word matching
            abs_correct = absolute_word_match(pred_, label_)

            with open('Results__Report_epoch{}.txt'.format(epoch), 'w') as fout2:
                fout2.write(report)


            with open('results.txt', 'w', encoding = 'utf-8') as fout:
                for x,y in zip(pred_, label_):
                    fout.write("True: {}".format(y))
                    fout.write("\n")
                    fout.write("Pred: {}".format(x))
                    fout.write("\n\n")
            print("Accuracy: %.4f" % accuracy)
            print("F1 Micro Score: %.4f" % f1_micro)
            print("F1 Macro Score: %.4f" % f1_macro)
            print("\n")

            ################################################JSON Dumps
            metrics1['epoch'].append(epoch)
            metrics1['accuracy'].append(accuracy)
            metrics1['train_loss'].append(train_loss)
            metrics1['valid_loss'].append(valid_loss)
            metrics1['total_wer'].append(total_wer)
            metrics1['f1_micro'].append(f1_micro)
            metrics1['f1_macro'].append(f1_macro)
            metrics1['absolute_word_correct'].append(abs_correct)

            json.dump( metrics1, open( "metrics(general).json", 'w' ) )

            metrics2['epoch'].append(epoch)
            metrics2['report'].append(report)

            json.dump( metrics2, open( "metrics(report).json", 'w' ) )


            print("End of Epoch {}".format(epoch))


            print(pred_[0])

            print("\n\n")
            return pred_[0]


    metrics1 = {
    'epoch': [],
    'accuracy': [],
    'train_loss': [],
    'valid_loss': [],
    'total_wer': [],
    'f1_micro': [],
    'f1_macro': [],
    'absolute_word_correct': [],
    }

    metrics2 = {
    'epoch': [],
    'report': [],
    }
    
    return validate(metrics1,metrics2,file_namelist)
##########################################################
@app.route("/upload_predict",methods = ["GET", "POST"])
def upload_predict():
    if request.method =="POST":
        image_file  = request.files["image"]
        if image_file:
            image_location = os.path.join(
                UPLOAD_FOLDER,
                image_file.filename
            )
            image_file.save(image_location)
            print(image_location)
            pred = img_path(image_location)
           # print(pred)
            return render_template("index.html",prediction=pred , image_loc = image_file.filename)


    return render_template("index.html",prediction=0,image_loc = None)

if __name__ == "__main__":
    app.run(debug=True)
