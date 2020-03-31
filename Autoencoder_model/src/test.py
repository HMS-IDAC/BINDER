import matplotlib.pyplot as plt
from data_gen import *
from sklearn.metrics import roc_curve, roc_auc_score, auc
from skimage.transform import resize
from Autoencoder_models import *
import argparse
from utils import *

## set the GPU 
os.environ["CUDA_VISIBLE_DEVICES"]="0"

parser=argparse.ArgumentParser(description='Model Info.')

parser.add_argument('-t', '--t_path', dest = 'test_path', help = "Path to Test data.")
parser.add_argument('-d', '--d_name', dest = 'test_dataset_name', help = "Name of test data = BINDER / PUBPEER / MFND ")
parser.add_argument('-r', '--r_path', dest = 'results_path', help = "Path to saving model results", default = './Model_results/')
parser.add_argument('-bs', '--batch_size', dest = 'batch_size', help = "Batch size to use", default = 2)
parser.add_argument('-n_channels', '--number_channels', dest = 'n_channels', help = "Number of channels, = 3 for RGB; = 1 for gray", default = 1)
parser.add_argument('--input_weights_path', dest = 'input_weights_path', help = 'path to load model weights')
parser.add_argument('--n_random_sel', dest = 'n_random_sel', help = 'number of repetitions', default = 1)

arguments = parser.parse_args()

if not arguments.test_path:   # if folder is not specified
	parser.error('Error: path to Test data must be specified. Pass --t_path or -t to command line')

if not arguments.input_weights_path:   # if folder is not specified
	parser.error('Error: path to Model weights to load must be specified. Pass --input_weights_path command line')

print('Setting path variables')

test_path = arguments.test_path
results_path = arguments.results_path
weights_model = arguments.input_weights_path
batch_size = arguments.batch_size
n_channels = arguments.n_channels
test_dataset_name = arguments.test_dataset_name
n_random_sel = arguments.n_random_sel

base_weights = './weights/Autoencoder_base_pretrained_COCO.hdf5' # Autoencoder pretrained on COCO to load into base encoder network
   
model_test= Autoencoder_top(pretrained_weights=True, weights_path=weights_model, base_weights=base_weights)

## Random selection
test_generator=test_gen(dir_path=test_path, test_dataset_name=test_dataset_name,neg_selection='random',batch_size=batch_size,n_channels=1)
data_gen=test_generator.select_generator()

true_labels=[]# to save results
pred_labels=[]

Y = np.empty([2,int(n_random_sel*len(os.listdir(test_path))),batch_size,128,128,n_channels]) # store image pairs
for i in range(int(n_random_sel*len(os.listdir(test_path)))):
    X=next(data_gen)
    true_labels.extend(X[1]['l2_norm_d'])
    predict=model_test.predict_on_batch(X[0])
    Y[0,i,] = X[0]['Input_l']
    Y[1,i,] = X[0]['Input_r']
    pred_labels.extend(predict[1])
true_labels=np.array(true_labels)
pred_labels=np.array(pred_labels)

print('\n Printing ROC & Prescision & recall metrics for random sel... \n')
        
ROC=f_acc_ROC(true_labels,pred_labels)
print('\n')
ROC_pres=f_acc_pres_recall(true_labels,pred_labels)
print('\n')

s = lambda same : same == 1
d = lambda diff : diff == 0
same_d=pred_labels[s(np.array(true_labels))]
diff_d=pred_labels[d(np.array(true_labels))]

plt.figure()
plt.hist(same_d, bins=30 ,alpha=0.5)
plt.hist(diff_d, bins=30,alpha=0.5)
plt.legend(['same','diff'])
plt.xlabel('distance')
plt.ylabel('Number of image pairs')
plt.title('Random Selection: {}'.format(test_dataset_name))
plt.show()
plt.savefig(results_path + 'random_selection.png')

plt.figure()
plt.plot(ROC[1][:-1],ROC[0][:-1])
plt.xlabel('Recall')
plt.ylabel('precision')
plt.title('Random: {}'.format(test_dataset_name))
plt.show()
plt.savefig(results_path + 'random_selection_precisionrecall.png')

from sklearn.metrics import roc_curve, roc_auc_score, auc
avg_pres_random=auc(ROC[1][:-1],ROC[0][:-1])
# calculate roc curve
fpr, tpr, thresholds = roc_curve(true_labels,  1/(pred_labels+sys.float_info.epsilon))
# calculate AUC
auc_random = roc_auc_score(true_labels,  1/(pred_labels+sys.float_info.epsilon))

plt.figure()
plt.plot(fpr,tpr,np.arange(0.,1,0.01),np.arange(0.,1,0.01))
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('Random: {}'.format(test_dataset_name))
plt.show()
plt.savefig(results_path + 'random_selection_ROC.png')

print('AUC Area Under the ROC urve:' + str(auc_random))
print('AP Average Precision:'+str(avg_pres_random) +'\n')

pred_labels=[]
true_labels=[]
## Hard Negative selection
test_generator_hardsel = test_gen(dir_path=test_path, test_dataset_name=test_dataset_name,neg_selection='hardneg',batch_size=batch_size,n_channels=1)
data_gen_hardneg = test_generator_hardsel.select_generator()
X_hardneg = next(data_gen_hardneg)
predict_hardneg = model_test.predict_on_batch(X_hardneg[0])
l2, hard_neg = l2dist_matrix(X[1]['l2_norm_d'],predict_hardneg[0])
same_dist = np.diag(l2)

true_labels=np.concatenate((np.ones([len(same_dist)]),np.zeros([len(hard_neg)])))
pred_labels=np.concatenate((same_dist,hard_neg))
print(true_labels.shape,pred_labels.shape, 'shape')
print('\n Printing ROC for hard neg \n')
ROC_hn=f_acc_ROC(true_labels,pred_labels)
print('\n')
#ROC_pres_hn=f_acc_pres_recall(true_labels,pred_labels)
#print('\n')

plt.figure()
plt.hist(same_dist, bins=30, alpha=0.6)
plt.hist(hard_neg, bins=30, alpha=0.5)
plt.legend(['same','hard_diff'])
plt.xlabel('distance')
plt.ylabel('Number of image pairs')
plt.title('Hardneg: {}'.format(test_dataset_name))
plt.show()
plt.savefig(results_path + 'hardneg_selection.png')

plt.figure()
plt.plot(ROC_hn[1][:-1],ROC_hn[0][:-1])
plt.xlabel('Recall')
plt.ylabel('precision')
plt.title('Hardneg: {}'.format(test_dataset_name))
plt.show()
plt.savefig(results_path + 'hardneg_selection_precisionrecall.png')


avg_pres_hn=auc(ROC_hn[1][:-1],ROC_hn[0][:-1])
# calculate roc curve
fpr, tpr, thresholds = roc_curve(true_labels,  1/(pred_labels+sys.float_info.epsilon))
# calculate AUC
auc_hn = roc_auc_score(true_labels,  1/(pred_labels+sys.float_info.epsilon))

plt.figure()
plt.plot(fpr,tpr,np.arange(0.,1,0.01),np.arange(0.,1,0.01))
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('Hardneg: {}'.format(test_dataset_name))
plt.show()
plt.savefig(results_path + 'hardneg_selection_ROC.png')

print('\nAUC Area Under the ROC curve:' + str(auc_hn))
print('AP Average Precision:'+str(avg_pres_hn) + '\n')

l2_matrix=np.array(l2).copy()

l2_diagonal=np.diag(l2)

threshold_low=input('Enter the lower bound threshold: ')
find_closestdiff = lambda x : x <= float(threshold_low)

## add offset to diagonal elements
np.fill_diagonal(l2_matrix,l2_diagonal+10)

index_diff=np.where(find_closestdiff(l2_matrix)==True)

print('\n Showing the closest different')
fig1 = plt.figure()
splt_n1=len(index_diff[0])

for i, (row,col) in enumerate(list(zip(index_diff[0],index_diff[1]))):
    
    ax1=fig1.add_subplot(2, splt_n1, i+1)
    ax1.imshow(np.squeeze(X_hardneg[0]['Input_l'][row,]))
    ax1=fig1.add_subplot(2, splt_n1, splt_n1+i+1)
    ax1.imshow(np.squeeze(X_hardneg[0]['Input_r'][col,]))
    
plt.show()


threshold_high=input('Enter the upper bound threshold: ')
find_farthestsame = lambda x : x >= float(threshold_high)
index_same=np.where(find_farthestsame(l2_diagonal)==True)


print('\n Showing the Farthest Same' )
fig2 = plt.figure()
splt_n2=len(index_same[0])

for i,row in enumerate(index_same[0]):
    
    ax2=fig2.add_subplot(2, splt_n2, i+1)
    ax2.imshow(np.squeeze(X_hardneg[0]['Input_l'][row,]))
    ax2=fig2.add_subplot(2, splt_n2, splt_n2+i+1)
    ax2.imshow(np.squeeze(X_hardneg[0]['Input_r'][row,]))
    
plt.show()