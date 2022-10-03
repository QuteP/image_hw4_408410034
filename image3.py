import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
import mpl_toolkits.axisartist as axisartist
from pic2str import image3j
import base64

def LaplacianEdge(img):
    height,width=img.shape
    new_img=np.zeros((height,width),dtype=np.float)
    padded_img=np.pad(img,((2,2),(2,2)),'reflect')
    core=np.array([[0,0,-1,0,0],[0,-1,-2,-1,0],[-1,-2,16,-2,-1],[0,-1,-2,-1,0],[0,0,-1,0,0]])
    for y in range(0,height):
        for x in range(0,width):
            new_img[y][x]=np.sum(padded_img[y:y+5,x:x+5]*core)
    # result_img=np.clip(img-new_img,0,255).astype(np.uint8)
    result_img=np.clip(new_img,0,255).astype(np.uint8)
    return result_img

def AverageSmooth(img,num=3):
    height,width=img.shape
    new_img=np.zeros((height,width),dtype=np.float)
    padded_img=np.pad(img,((num//2,num//2),(num//2,num//2)),'reflect')
    core=np.ones((num,num),dtype=np.float)
    for y in range(0,height):
        for x in range(0,width):
            new_img[y][x]=(np.sum(padded_img[y:y+num,x:x+num]*core))/(num*num)
    return np.clip(new_img,0,255).astype(np.uint8)

def SobelEdge(img):
    height,width=img.shape
    # img=AverageSmooth(img,num=3)
    x_img=np.zeros((height,width),dtype=np.float)
    y_img=np.zeros((height,width),dtype=np.float)
    pos_img=np.zeros((height,width),dtype=np.float)
    neg_img=np.zeros((height,width),dtype=np.float)
    x_padded_img=np.pad(img,((1,1),(1,1)),'reflect')
    y_padded_img=np.pad(img,((1,1),(1,1)),'reflect')
    neg_padded_img=np.pad(img,((1,1),(1,1)),'reflect')
    pos_padded_img=np.pad(img,((1,1),(1,1)),'reflect')
    x_core=np.array([[-1,-2,-1],[0,0,0],[1,2,1]])
    y_core=np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
    neg_core=np.array([[0,1,2],[-1,0,1],[-2,-1,0]])
    pos_core=np.array([[-2,-1,0],[-1,0,1],[0,1,2]])
    for h in range(0,height):
        for w in range(0,width):
            x_img[h][w]=abs(np.sum(x_padded_img[h:h+3,w:w+3]*x_core))
            y_img[h][w]=abs(np.sum(y_padded_img[h:h+3,w:w+3]*y_core))
            pos_img[h][w]=abs(np.sum(pos_padded_img[h:h+3,w:w+3]*pos_core))
            neg_img[h][w]=abs(np.sum(neg_padded_img[h:h+3,w:w+3]*neg_core))
    # result_img=x_img + y_img + pos_img + neg_img
    # result_img=pos_img + neg_img
    result_img=x_img + y_img
    return np.clip(result_img,0,255).astype(np.uint8)

def ReadByteImg(func):
    byte_data = base64.b64decode(func)
    img_buffer_numpy = np.frombuffer(byte_data, dtype=np.uint8)  # 将 图片字节码bytes  转换成一维的numpy数组 到缓存中
    return cv2.imdecode(img_buffer_numpy, cv2.IMREAD_UNCHANGED)

def PlotHisto(fig,data,pos_num):
    ax = axisartist.Subplot(fig,1,3,pos_num)
    fig.add_subplot(ax)
    ax.imshow(data,cmap='gray')
    ax.axis["bottom"].set_visible(False)
    ax.axis["left"].set_visible(False)
    ax.axis["top"].set_visible(False)
    ax.axis["right"].set_visible(False)
    plt.tight_layout()

def PlotBackground(i,title):
    ax = axisartist.Subplot(fig, 1,3,i)
    fig.add_subplot(ax)
    ax.axis["bottom"].set_visible(False)
    ax.axis["left"].set_visible(False)
    ax.axis["top"].set_visible(False)
    ax.axis["right"].set_visible(False)
    ax.annotate(title, xy=(0,0.5))
    plt.tight_layout()

image3=cv2.imread('./image3.jpg',cv2.IMREAD_UNCHANGED)
if type(image3)==type(None):
    image3=ReadByteImg(image3j)

fig = plt.figure()
img_lt=[image3]
for i,(img) in enumerate(img_lt):
    log_img=LaplacianEdge(img)
    sobel_img=SobelEdge(img)
    PlotHisto(fig,img,1)
    PlotHisto(fig,log_img,2)
    PlotHisto(fig,sobel_img,3)

plt.tight_layout()
plt.show()