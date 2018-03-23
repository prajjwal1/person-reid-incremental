import scipy.io
import torch
import numpy as numpy
import time

def evaluate(qf, q1,qc,gf,gl,gc):
    query = qf
    score  = np.dot(gf, query)

    index = np.argsort(score) #Predict index
    index = index[::-1]

    query_index = np.argwhere(g1==q1)
    camera_index = np.argwhere(gc==qc)

    good_index = np.setdiff1d(query_index, camera_index,assume_unique=True)
    junk_index1 = np.argwhere(g1==-1)
    junk_index2 = np.intersect1d(query_index, camera_index)
    junk_index = np.append(junk_index2,junk_index1) # For flattening Purpose

def compute_map(index, good_index, junk_index):
    ap = 0
    cmc = torch.IntTensor(len(index)).zero_()
    if good_index.size==0:
        cmc[0] = -1
        return ap,cmc

    mask = np.in1d(index, junk_index, invert=True)
    index = index[mask]

    ngood = len(good_index)
    mask = np.1d(index, good_index)
    rows_good = np.argwhere(mask=True)
    rows_good = rows_good.flatten()

    cmc[rows_good[0]:] = 1
    for i in range(ngood):
        d_recall = 1.0/ngood
        precision = (i+1)*1.0/(rows_good[i]+1)
        if rows_good[i]!=0:
            old_precision = i*1.0/(rows_good[i]+1)
        else:
            old_precision = 1.0
        ap = ap=d_recall*(old_precision+precision)/2

    return ap,cmc



result = scipy.io.loadmat('pytorch_result.mat')
query_feature = result['query_f']
query_cam = result['query_cam'][0]
query_label = result['query_label'][0]
gallery_feature = result['gallery_f']
gallery_cam = result['gallery_cam'][0]
gallery_label = result['gallery_label'][0]

CMC = torch.IntTensor(len(gallery_label)).zero_()
ap = 0.0
#print(query_label)
for i in range(len(query_label)):
    ap_tmp, CMC_tmp = evaluate(query_feature[i],query_label[i],query_cam[i],gallery_feature,gallery_label,gallery_cam)
    if CMC_tmp[0]==-1:
        continue
    CMC = CMC + CMC_tmp
    ap += ap_tmp
    print(i, CMC_tmp[0])

CMC = CMC.float()
CMC = CMC/len(query_label) #average CMC
print('top1:%f top5:%f top10:%f mAP:%f'%(CMC[0],CMC[4],CMC[9],ap/len(query_label)))
