import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
import os
import sys
import cv2
from sklearn.metrics import roc_curve, auc
from skimage.feature import haar_like_feature_coord
from skimage.feature import draw_haar_like_feature
from skimage.feature import haar_like_feature
from skimage.transform import integral_image

feature_types=['type-2-x','type-2-y','type-3-x','type-3-y','type-4']
def vectorize(img):
    height ,width ,depth = img.shape
    return img.reshape((height*width*depth))

def plot_roc(score, num_pos, num_neg, save_dir, name):

    labels = np.zeros((num_pos+num_neg, ), dtype=np.float)
    labels[:num_pos] = 1

    fpr, tpr, threshold = metrics.roc_curve(labels, score)
    roc_auc = metrics.auc(fpr, tpr)

    plt.title('ROC:' + name)
    plt.plot(fpr, tpr, 'b', label='AUC = {:0.2f}'.format(roc_auc))
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.savefig(os.path.join(save_dir, name+'.jpg'))
    plt.close()

def plot_ROC_curve(post_f_nf, post_f_f,size):
    predictions = np.append(post_f_nf, post_f_f)
    temp1 = [0]*size
    temp2 = [1]*size
    actual = np.append(temp1,temp2)
    false_positive_rate, true_positive_rate, _ = roc_curve(actual, predictions)
    roc_auc = auc(false_positive_rate, true_positive_rate)
    plt.plot(false_positive_rate, true_positive_rate, 'b')
    plt.show()

def check_path(path):
    if not os.path.exists(path):
        print('Not found {}'.format(path))
        sys.exit(0)

def make_dir_if_not_exist(path):
    if not os.path.exists(path):
        os.makedirs(path)

def visualize_mean_cov(mu, cov, shape, save_dir, name):

    make_dir_if_not_exist(save_dir)

    mu_image = np.reshape(mu, shape)
    save_name = os.path.join(save_dir, name+'_mean.jpg')
    cv2.imwrite(save_name, mu_image)

    cov_image = np.reshape(np.sqrt(np.diag(cov)), shape)
    save_name = os.path.join(save_dir, name+'_diag_cov.jpg')
    cv2.imwrite(save_name, cov_image)

def haar_feature_descriptor(img,type,feature_coords=None,feature_type=None):
    integ_img=integral_image(img)
    if type==0:
        features=haar_like_feature(integ_img,0,0,img.shape[1],img.shape[0],feature_types)
    else:
        features=haar_like_feature(integ_img,0,0,img.shape[1],img.shape[0],feature_type=feature_type,feature_coord=feature_coords)
    #print(len(features))
    return features

def get_haar_accuracy_list(feature_stack):
    no_of_features=feature_stack.shape[1]
    accuracy_arr=[]
    for i in range(0,no_of_features):
        face_accuracy=np.sum(feature_stack[0:500,i]==1)
        non_face_accuracy=np.sum(feature_stack[500:1000,i]==-1)
        accuracy=(face_accuracy+non_face_accuracy)/1000
        accuracy_arr.append(accuracy)
    accuracy_arr=np.asarray(accuracy_arr)
    return accuracy_arr

def get_haar_coordinates(img):
    coord, feature_type =haar_like_feature_coord(img.shape[1],img.shape[0],feature_types)
    return coord, feature_type

def draw_and_plot_haar_features(img, feature_coords):

    fig, axs = plt.subplots(5, 2)
    for ax, feat_c in zip(np.ravel(axs), feature_coords):
        haar_feature = draw_haar_like_feature(img, 0, 0,img.shape[0],img.shape[1],[feat_c])
        ax.imshow(haar_feature)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle('After Boosting')
    plt.axis('off')
    plt.show()

def plotROG(face_prediction,nonface_prediction):
    fpr=[]
    tpr=[]

    for thresh in range(-10000,10000):
        face_arr=np.where(face_prediction>=thresh,1,-1)
        true_positives=np.sum(face_arr==1)
        false_negatives=face_arr.shape[0]-true_positives

        non_face_arr=np.where(nonface_prediction>=thresh,1,-1)
        true_negatives=np.sum(non_face_arr==-1)
        false_positives=non_face_arr.shape[0]-true_negatives

        if thresh==0:
            accuracy=(true_positives+true_negatives)/(face_arr.shape[0]+non_face_arr.shape[0])
            print("accuracy is "+str(accuracy))

        true_pos_rate=true_positives/(true_positives+false_negatives)
        false_pos_rate=false_positives/(false_positives+true_negatives)

        tpr.append(true_pos_rate)
        fpr.append(false_pos_rate)

    plt.title('ROC curve')
    plt.plot(fpr, tpr, 'b')
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()

