import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageOps, ImageDraw
from skimage import draw
from tqdm import tqdm

ax = plt.subplot()
n = 200

########################################################################################################################
#Load, Crop and prep initial image.
img = Image.open("girl.jpg")
img = ImageOps.grayscale(img)
h = min(img.size)
canvas = np.full((h,h),255)
img_arr = np.array(img)[0:h][0:h]
ax.set_xlim(0,h)
ax.set_ylim(h,0)
X = np.linspace(0, 2 * np.pi, n)
ax.plot(h/2*np.cos(X)+h/2,h/2*np.sin(X)+h/2, 'rx')
########################################################################################################################

def ntocord(node):
    """
    :param node: Enumerated node on circle.
    :return: Cordinates for said node.
    """
    return (h/2*np.cos(X[node])+h/2, h/2*np.sin(X[node])+h/2)


def avgdark(node1,node2): #node in 0:n
    """
    node1, node2 are the numerated points on the circle.
    Returns average grayscalevalue of pixels on the line from node1 to node2.
    """
    x1, y1 = ntocord(node1)
    x2, y2 = ntocord(node2)

    avggray = 0
    pixls = draw.line(int(x1),int(y1),int(x2),int(y2))
    for i,j in zip(pixls[0],pixls[1]):
        gray = img_arr[j-1][i-1]
        avggray+=gray
    return avggray/len(pixls[0])


def repaint(node1,node2,matrix,white=True):
    """
    :param node1:
    :param node2:
    :param matrix: image matrix to draw on.
    :param white: boolean, white if true, black if false
    :return: void, but paints line between node1 and node2 white on image.
    """
    x1, y1 = ntocord(node1)
    x2, y2 = ntocord(node2)
    pixls = draw.line(int(x1),int(y1),int(x2),int(y2))
    for i,j in zip(pixls[0],pixls[1]):
        if white:
            matrix[j-1][i-1] = 255
        else:
            matrix[j - 1][i - 1] = 0


def darkestnode(node,gap):
    """
    :param node:  node to search from.
    :param gap:  Nearest allowed connecting neighbour.
    :return:   node for which connecting path is darkest.
    """
    nyx=255
    renode=None
    for i in range(n-2*gap):
        nextnode = (node+gap+i)%n
        tnyx=avgdark(node,nextnode)
        if tnyx<nyx:
            nyx=tnyx
            renode=nextnode
    return renode

def nodeseq(snode,iter,paint=True):
    """
    :param snode: startingnode
    :param iter: iterations
    :param paint: paint strings on canvas
    :return: sequence of nodes to obtain string portrait
    """
    seq=[snode]
    for i in tqdm(range(iter)): #tqdm for progress bar
        seq+=[darkestnode(seq[i],round(n*0.05))]
        repaint(seq[i],seq[i+1],img_arr)
        if paint:
            repaint(seq[i],seq[i+1],canvas,white=False)
    return seq

print(nodeseq(0,3000))

img = Image.fromarray(canvas)
img.show()

ax.imshow(img_arr,cmap=plt.get_cmap('gray'),vmin=0,vmax=255)
plt.show()