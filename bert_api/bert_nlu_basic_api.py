# -*- coding: utf-8 -*-
"""
Created on Sat Jan 11 15:12:18 2020

@author: mwahdan
"""

from flask import Flask, jsonify, request
from vectorizers.bert_vectorizer import BERTVectorizer
from models.joint_bert import JointBertModel
from utils import convert_to_slots
from tensorflow.python.keras.backend import set_session
import tensorflow as tf
import pickle
import argparse
import os

import tensorflow as tf
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Input, Dense, Multiply, TimeDistributed
from models.nlu_model import NLUModel
import tensorflow_hub as hub
import numpy as np
import os
import json


class JointBertModel1(NLUModel):

    def __init__(self,intents_num, bert_hub_path, num_bert_fine_tune_layers=10,
                 is_bert=True):
        #self.slots_num = slots_num
        self.intents_num = intents_num
        self.bert_hub_path = bert_hub_path
        self.num_bert_fine_tune_layers = num_bert_fine_tune_layers
        self.is_bert = is_bert
        
        self.model_params = {
                'intents_num': intents_num,
                'bert_hub_path': bert_hub_path,
                'num_bert_fine_tune_layers': num_bert_fine_tune_layers,
                'is_bert': is_bert
                }
        
        self.build_model()
        self.compile_model()
        
        
    def compile_model(self):
        # Instead of `using categorical_crossentropy`, 
        # we use `sparse_categorical_crossentropy`, which does expect integer targets.
        
        optimizer = tf.keras.optimizers.Adam(lr=5e-5)#0.001)

        losses = {
        	'intent_classifier': 'sparse_categorical_crossentropy',
        }
        loss_weights = {'intent_classifier': 1.0}
        metrics = {'intent_classifier': 'acc'}
        self.model.compile(optimizer=optimizer, loss=losses, loss_weights=loss_weights, metrics=metrics)
        self.model.summary()
        

    def build_model(self):
        in_id = Input(shape=(None,), name='input_word_ids', dtype=tf.int32)
        in_mask = Input(shape=(None,), name='input_mask', dtype=tf.int32)
        in_segment = Input(shape=(None,), name='input_type_ids', dtype=tf.int32)
        #in_valid_positions = Input(shape=(None, self.slots_num), name='valid_positions')
        bert_inputs = [in_id, in_mask, in_segment]
        inputs = bert_inputs
        
        if self.is_bert:
            name = 'BertLayer'
        else:
            name = 'AlbertLayer'
        bert_pooled_output, bert_sequence_output = hub.KerasLayer(self.bert_hub_path,
                              trainable=True, name=name)(bert_inputs)
        
        intents_fc = Dense(self.intents_num, activation='softmax', name='intent_classifier')(bert_pooled_output)
        
        self.model = Model(inputs=inputs, outputs=intents_fc)

        
    def fit(self, X, Y, validation_data=None, epochs=5, batch_size=32):
        """
        X: batch of [input_ids, input_mask, segment_ids, valid_positions]
        """
        X = (X[0], X[1], X[2])
        if validation_data is not None:
            print("INSIDE")
            X_val, Y_val = validation_data
            validation_data = ((X_val[0], X_val[1], X_val[2]), Y_val)
        
        history = self.model.fit(X, Y, validation_data=validation_data, 
                                 epochs=epochs, batch_size=batch_size)
        #self.visualize_metric(history.history, 'slots_tagger_loss')
        #self.visualize_metric(history.history, 'intent_classifier_loss')
        #self.visualize_metric(history.history, 'loss')
        #self.visualize_metric(history.history, 'intent_classifier_acc')
        
        
    def prepare_valid_positions(self, in_valid_positions):
        in_valid_positions = np.expand_dims(in_valid_positions, axis=2)
        in_valid_positions = np.tile(in_valid_positions, (1, 1, self.slots_num))
        return in_valid_positions
    
                
        
    def predict_slots_intent(self, x, slots_vectorizer, intent_vectorizer, remove_start_end=True,
                             include_intent_prob=False):
        valid_positions = x[3]
        x = (x[0], x[1], x[2], self.prepare_valid_positions(valid_positions))
        y_slots, y_intent = self.predict(x)
        slots = slots_vectorizer.inverse_transform(y_slots, valid_positions)
        if remove_start_end:
            slots = [x[1:-1] for x in slots]
            
        if not include_intent_prob:
            intents = np.array([intent_vectorizer.inverse_transform([np.argmax(i)])[0] for i in y_intent])
        else:
            intents = np.array([(intent_vectorizer.inverse_transform([np.argmax(i)])[0], round(float(np.max(i)), 4)) for i in y_intent])
        return slots, intents
    

    def save(self, model_path):
        with open(os.path.join(model_path, 'params.json'), 'w') as json_file:
            json.dump(self.model_params, json_file)
        self.model.save(os.path.join(model_path, 'joint_bert_model.h5'))
        
        
    def load(load_folder_path):
        with open(os.path.join(load_folder_path, 'params.json'), 'r') as json_file:
            model_params = json.load(json_file)
            
        #slots_num = model_params['slots_num'] 
        intents_num = model_params['intents_num']
        bert_hub_path = model_params['bert_hub_path']
        num_bert_fine_tune_layers = model_params['num_bert_fine_tune_layers']
        is_bert = model_params['is_bert']
            
        new_model = JointBertModel1(intents_num, bert_hub_path, num_bert_fine_tune_layers, is_bert)
        new_model.model.load_weights(os.path.join(load_folder_path,'joint_bert_model.h5'))
        return new_model
    
# Create app
#app = Flask(__name__)
dict1 =     {"0":"Bundle_Catalog_items","1":"Get_Approvals","2":"Get_PTO","3":"Get_holiday_calender","4":"Get_incidents","5":"Get_knowledge_base","6":"Open_ticket","7":"Request_PTO","8":"Reset_password","9":"Update_ticket","10":"affirm","11":"deny","12":"goodbye","13":"greet","14":"high_intent","15":"live_agent","16":"low_intent","17":"medium_intent","18":"stop","19":"welcome"}
bert_vectorizer = BERTVectorizer(True, "https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/1")
model = JointBertModel1.load("./models")
import time
app = Flask(__name__) 

@app.route('/intent', methods = ['POST']) 
def predict():
    data = request.get_json()["utterance"]
    print("---------------")
    print(data)
    a = time.time()
    #tokens = utterance.split()
    #print(utterance)
    input_ids, input_mask, segment_ids, data_sequence_lengths = bert_vectorizer.transform([data])
    predicted_intents = model.predict([input_ids, input_mask, segment_ids])
    b = time.time()
    intent = dict1[str(np.argmax(predicted_intents))]
    confidence = np.amax(predicted_intents)
    print("TIme taken : ",str(b-a))
    return jsonify({'intent': intent,"confidence":str(confidence)})


if __name__ == '__main__':
    app.run(host='0.0.0.0',debug = True,port=12922) 


