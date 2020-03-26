import numpy as np
import math
import utilities

class Adaboost:
    def __init__(self, training_size, haar_features_stack):
        self.weights=np.full(training_size, (1/training_size), dtype=np.float32)
        self.alpha_list=[]
        self.haar_index=[]
        self.haar_stack=haar_features_stack

    def calculate_weighted_error(self,face_training_haar,non_face_training_haar):
        haar_binary_face=(face_training_haar!=1)
        haar_binary_nonface=(non_face_training_haar!=-1)
        haar_binary_training=np.vstack((haar_binary_face,haar_binary_nonface))
        self.weighted_errors=np.matmul(self.weights,haar_binary_training)
        index_sorted=np.argsort(self.weighted_errors)
        for index in index_sorted:
            if index not in self.haar_index:
                break

        self.current_weighted_error=self.weighted_errors[index]
        self.haar_index.append(index)
        self.current_haar=self.haar_stack[:,index]
        #self.haar_stack=np.delete(self.haar_stack,index,axis=1)


    def calculate_alpha(self):
        self.alpha=0.5*math.log((1-self.current_weighted_error)/self.current_weighted_error)
        self.alpha_list.append(self.alpha)

    def update_weight(self,y):
        #self.haar_training=np.concatenate((training_face, training_nonface))
        temp=-1*self.alpha*np.multiply(y,self.current_haar)
        exponential=np.exp(temp)
        Z=np.matmul(self.weights,exponential)
        self.weights=(1/Z)*np.multiply(self.weights,exponential)

    def test(self,test_data_face,test_data_nonface,feature_coords,feature_type):
        prediction_face_data=[]
        prediction_non_face_data=[]

        for face_img in test_data_face:
            prediction_face_data.append(np.matmul(self.alpha_list,utilities.haar_feature_descriptor(face_img,1,feature_coords,feature_type)))
        for non_face_img in test_data_nonface:
            prediction_non_face_data.append(np.matmul(self.alpha_list,utilities.haar_feature_descriptor(non_face_img,1,feature_coords,feature_type)))

        prediction_face_data=np.asarray(prediction_face_data)
        prediction_non_face_data=np.asarray(prediction_non_face_data)

        return prediction_face_data, prediction_non_face_data


