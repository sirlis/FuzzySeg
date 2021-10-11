import os
import numpy as np
from sklearn.metrics import confusion_matrix # for iou
from PIL import Image
from pdf2image import convert_from_path

def jaccard_iou(y_pred, y_true, labels):
    current = confusion_matrix(y_true, y_pred, labels=labels)
    # compute mean iou
    intersection = np.diag(current)
    ground_truth_set = current.sum(axis=1)
    predicted_set = current.sum(axis=0)
    union = ground_truth_set + predicted_set - intersection
    IoU = intersection / union.astype(np.float32)
    return np.mean(IoU)

def pearson(y_pred, y_true):
#只计算两者共同有的
    same = 0
    for i in y_pred:
        if i in y_true:
            same +=1
    n = same
    #分别求p，q的和
    sumx = sum([y_pred[i] for i in range(n)])
    sumy = sum([y_true[i] for i in range(n)])
    #分别求出p，q的平方和
    sumxsq = sum([y_pred[i]**2 for i in range(n)])
    sumysq = sum([y_true[i]**2 for i in range(n)])
    #求出p，q的乘积和
    sumxy = sum([y_pred[i]*y_true[i] for i in range(n)])
    # print sumxy
    #求出pearson相关系数
    up = sumxy - sumx*sumy/n
    down = ((sumxsq - pow(sumxsq,2)/n)*(sumysq - pow(sumysq,2)/n))**.5
    #若down为零则不能计算，return 0
    if down == 0 :return 0
    r = up/down
    return r

def yule(fpA, fpB):
    if not len(fpA) == len(fpB):
        print('WARNING! lenth mismatch!')
        return
    # bit count
    onlyA, onlyB, bothAB, neitherAB = 0,0,0,0
    for i in range(len(fpA)):
        if fpA[i] == 1:
            if fpB[i] == 1:
                bothAB = bothAB + 1
            else:
                onlyA = onlyA + 1
        elif fpA[i] == 0:
            if fpB[i] == 1:
                onlyB = onlyB + 1
            else:
                neitherAB = neitherAB + 1
    yule = float(bothAB * neitherAB - onlyA * onlyB)
    yule /= float(bothAB * neitherAB + onlyA * onlyB)
    return yule

def threth_seg(imgt, thre=100):
    table = []
    imgt1 = np.asarray(imgt)
    for i in range(imgt1.shape[0]):
        for j in range(imgt1.shape[1]):
            if  imgt1[i, j]  <  thre:
                table.append(0)
            else:
                table.append(1)
    size = imgt.size
    return np.array(table, dtype=np.float64), size




if __name__=='__main__':
        
    path = os.getcwd()

    ### convert pdf to jpg
    # pages = convert_from_path('F:/Papers/20210706_FuzzySeg/code_helix/extra/Fig3b-eps-converted-to.pdf', 500)
    # for page in pages:
    #     page.save('Fig3b.jpg', 'JPEG')
    # pages = convert_from_path('F:/Papers/20210706_FuzzySeg/code_helix/extra/Fig3c-eps-converted-to.pdf', 500)
    # for page in pages:
    #     page.save('Fig3c.jpg', 'JPEG')
    # pages = convert_from_path('F:/Papers/20210706_FuzzySeg/code_helix/extra/Fig3d-eps-converted-to.pdf', 500)
    # for page in pages:
    #     page.save('Fig3d.jpg', 'JPEG')
    # pages = convert_from_path('F:/Papers/20210706_FuzzySeg/code_helix/extra/Fig3e-eps-converted-to.pdf', 500)
    # for page in pages:
    #     page.save('Fig3e.jpg', 'JPEG')
    # pages = convert_from_path('F:/Papers/20210706_FuzzySeg/code_helix/extra/Fig3f-eps-converted-to.pdf', 500)
    # for page in pages:
    #     page.save('Fig3f.jpg', 'JPEG')


    ### threthold segmentation from jpg
    threshold = 100
    table = []
    imgt = Image.open(os.path.join(path,'Fig3b.jpg')).convert("L")
    table,size = threth_seg(imgt,threshold)

    imkm = Image.open(os.path.join(path,'Fig3c.jpg')).convert("L")
    imkm = imkm.resize(size,Image.ANTIALIAS)
    tabelkm,_ = threth_seg(imkm,threshold)

    imgc = Image.open(os.path.join(path,'Fig3d.jpg')).convert("L")
    imgc = imgc.resize(size,Image.ANTIALIAS)
    tabelgc,_ = threth_seg(imgc,threshold)

    imssc = Image.open(os.path.join(path,'Fig3e.jpg')).convert("L")
    imssc = imssc.resize(size,Image.ANTIALIAS)
    tabelssc,_ = threth_seg(imssc,threshold)

    imfpsc = Image.open(os.path.join(path,'Fig3f.jpg')).convert("L")
    imfpsc = imfpsc.resize(size,Image.ANTIALIAS)
    tabelfpsc,_ = threth_seg(imfpsc,threshold)

    ### calculating iou (jaccard)
    # iou_km = jaccard_iou(tabelkm, table, [0,1])
    # iou_gc = jaccard_iou(tabelgc, table, [0,1])
    # iou_ssc = jaccard_iou(tabelssc, table, [0,1])
    # iou_fpsc = jaccard_iou(tabelfpsc, table, [0,1])
    # print(iou_km)
    # print(iou_gc)
    # print(iou_ssc)
    # print(iou_fpsc)

    ### calculating pearson similarity coefficient
    # pea_km = pearson(tabelkm, table)
    # pea_gc = pearson(tabelgc, table)
    # pea_ssc = pearson(tabelssc, table)
    # pea_fpsc = pearson(tabelfpsc, table)
    # print(pea_km)
    # print(pea_gc)
    # print(pea_ssc)
    # print(pea_fpsc)

    ### calculating yule similarity coefficient
    yule_km = yule(tabelkm, table)
    yule_gc = yule(tabelgc, table)
    yule_ssc = yule(tabelssc, table)
    yule_fpsc = yule(tabelfpsc, table)
    print(yule_km)
    print(yule_gc)
    print(yule_ssc)
    print(yule_fpsc)

