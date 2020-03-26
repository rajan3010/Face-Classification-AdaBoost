import numpy as np
import dataloadaer
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import utilities
import adaboost

#dim = 100
#test_size = 100
#train_size = 1000

def vectorizeandstack(data):
    face_data = np.empty([0, 1200], dtype=np.float32)
    for img in data:
        data=np.array(img).flatten()
        face_data=np.vstack((face_data,data))

    return face_data

def get_max_value_index(arr,n):
    arr=np.argsort(-arr)[:n]
    return arr

def generate_weak_classifier(data_face, data_nonface):
    haar_features_stack_face = np.empty([0,78134], dtype=np.uint8)
    haar_features_stack_nonface = np.empty([0, 78134], dtype=np.uint8)
    for img in data_face:
        haar_features_face=utilities.haar_feature_descriptor(img,0)
        haar_features_stack_face=np.vstack((haar_features_stack_face, haar_features_face))

    for image in data_nonface:
        haar_features_nonface = utilities.haar_feature_descriptor(image,0)
        #haar_feature_coord_nonface = utilities.get_haar_coordinates(image)
        haar_features_stack_nonface = np.vstack((haar_features_stack_nonface, haar_features_nonface))

    haar_feature_coord, haar_feature_type=utilities.get_haar_coordinates(data_face[0])
    haar_features_stack_face=np.where(haar_features_stack_face>=0,1,-1) #Generating values in the set {+1, -1} for +1 face, -1 non-face
    haar_features_stack_nonface=np.where(haar_features_stack_nonface>=0,1,-1)
    haar_features_stack=np.vstack((haar_features_stack_face,haar_features_stack_nonface))
    accuracy_stack=utilities.get_haar_accuracy_list(haar_features_stack)
    max_index_arr=get_max_value_index(accuracy_stack,10) #get the index of features giving the maximum accuracy
    max_index_feature_coords=np.take(haar_feature_coord,max_index_arr)
    print(accuracy_stack[max_index_arr[0]])
    utilities.draw_and_plot_haar_features(data_face[0],max_index_feature_coords)
    exit()
    return haar_feature_coord, haar_feature_type, max_index_arr, haar_features_stack, haar_features_stack_face,haar_features_stack_nonface

def main():

    training_data_face, training_data_nonface, testing_data_face, testing_data_nonface = dataloadaer.loadandwrite()
    training_data=training_data_face+training_data_nonface
    training_output_face=np.full(500,1)
    training_output_nonface=np.full(500,-1)
    training_output=np.concatenate((training_output_face,training_output_nonface))
    feature_coords, feature_type, max_index_arr, haar_feature_stack, haar_features_stack_face, haar_features_stack_nonface= generate_weak_classifier(training_data_face, training_data_nonface)
    model=adaboost.Adaboost(len(training_data), haar_feature_stack)

    for t in range(0,20):
        model.calculate_weighted_error(haar_features_stack_face, haar_features_stack_nonface)
        model.calculate_alpha()
        model.update_weight(training_output)

    boosted_feature_coords_display=np.take(feature_coords,model.haar_index[0:10])
    boosted_feature_type=np.take(feature_type,model.haar_index)
    boosted_feature_coords=np.take(feature_coords,model.haar_index)
    utilities.draw_and_plot_haar_features(training_data_face[0],boosted_feature_coords_display)
    face_predict, non_face_predict=model.test(testing_data_face,testing_data_nonface,boosted_feature_coords,boosted_feature_type)
    utilities.plotROG(face_predict, non_face_predict)


main()
