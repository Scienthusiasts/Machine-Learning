import numpy as np     
import matplotlib.pyplot as plt






'''读取数据集并将类别转化为整型'''
# def readFile(path):
#     file = open(path) # 读取文件
#     # 切分数据的不同维度
#     data = np.array([line.split('\t') for line in file.readlines()])
#     # 读取数据不同维度下可能的取值
#     # set()创建一个无序且不重复元素集和
#     label = [list(set(data[:, i])) for i in range(data.shape[1])]
#     axis = range(data.shape[1]) # 建立索引
#     # 将类别转化为字典的key,类别的序号为value
#     label_dic = [dict(zip(label[i], axis)) for i in range(data.shape[1])]
#     for i in range(data.shape[0]):
#         for j in range(data.shape[1]):
#             # 通过字典的value将类别转化为整型
#             data[i, j] = label_dic[j][data[i, j]]
#     # 返回处理后的数据与标签对应的字符串
#     return data[:, :-1].astype(np.int), data[:, -1].astype(np.int), label




'''读取数据集并转化为np.array'''
def readFile(path):
    file = open(path) # 读取文件
    cls_value = file.readline().split('\t')[:-1]
    # 切分数据的不同维度
    data = np.array([line.split('\t') for line in file.readlines()])
    return data[:, :-1], data[:, -1], np.array(cls_value)





# 递归获取树的叶子结点数(深度优先遍历)
def getNumLeaves(DTree):
    num_leaves = 0
    root = list(DTree.keys())[0]
    son_node = DTree[root]
    for key in son_node.keys():
        if type(son_node[key]).__name__ == 'dict':
            # 递归计算子节点
            num_leaves += getNumLeaves(son_node[key])
        else: 
            num_leaves += 1
    return num_leaves



# 递归获取树的深度(深度优先遍历)
def getTreeDepth(DTree):
    max_depth = 0
    root = list(DTree.keys())[0]
    son_node = DTree[root]
    # 返回当前层的最大深度
    for key in son_node.keys():
        if type(son_node[key]).__name__ == 'dict':
            # 递归计算子节点的深度
            this_depth = getTreeDepth(son_node[key])
        else: 
            # 如果子节点是叶子结点，则子节点的深度为1
            this_depth = 1
        # 子树的深度取决于最大的子节点的深度
        if this_depth > max_depth:
            max_depth = this_depth
    # 树的深度 = 1(根节点) + 子节点的深度
    return max_depth + 1





decisionNode = {"boxstyle":"sawtooth", "fc":"0.9"}
leafNode = {"boxstyle":"round4", "fc":"0.6"}
arrow_args = {"arrowstyle":"<-"}


def plotMidText(cntrPt, parentPt, txtString):
    xMid = (parentPt[0]-cntrPt[0])/2.0 + cntrPt[0]
    yMid = (parentPt[1]-cntrPt[1])/2.0 + cntrPt[1]
    createPlot.ax1.text(xMid, yMid, txtString, va="center", ha="center", rotation=30)


def plotNode(nodeTxt, centerPt, parentPt, nodeType):
    createPlot.ax1.annotate(nodeTxt, xy=parentPt, xycoords='axes fraction',
    xytext=centerPt, textcoords='axes fraction', 
    va='center', ha='center', bbox=nodeType, arrowprops=arrow_args
    )


def plotTree(DTree, parentPt, nodeTxt):#if the first key tells you what feat was split on
    numLeafs = getNumLeaves(DTree)  #this determines the x width of this tree
    depth = getTreeDepth(DTree)
    firstStr = list(DTree.keys())[0]     #the text label for this node should be this
    cntrPt = (plotTree.xOff + (1.0 + float(numLeafs))/2.0/plotTree.totalW, plotTree.yOff)
    plotMidText(cntrPt, parentPt, nodeTxt)
    plotNode(firstStr, cntrPt, parentPt, decisionNode)
    secondDict = DTree[firstStr]
    plotTree.yOff = plotTree.yOff - 1.0/plotTree.totalD
    for key in secondDict.keys():
        if type(secondDict[key]).__name__=='dict':#test to see if the nodes are dictonaires, if not they are leaf nodes   
            plotTree(secondDict[key],cntrPt,str(key))        #recursion
        else:   #it's a leaf node print the leaf node
            plotTree.xOff = plotTree.xOff + 1.0/plotTree.totalW
            plotNode(secondDict[key], (plotTree.xOff, plotTree.yOff), cntrPt, leafNode)
            plotMidText((plotTree.xOff, plotTree.yOff), cntrPt, str(key))
    plotTree.yOff = plotTree.yOff + 1.0/plotTree.totalD
#if you do get a dictonary you know it's a tree, and the first element will be another dict



# 决策树节点绘制模块:
def createPlot(DTree):
    fig = plt.figure(1, facecolor='white', dpi=10, figsize=(30, 15))
    fig.clf()
    axprops = dict(xticks=[], yticks=[])
    createPlot.ax1 = plt.subplot(111, frameon=False, **axprops)    #no ticks
    #createPlot.ax1 = plt.subplot(111, frameon=False) #ticks for demo puropses 
    plotTree.totalW = float(getNumLeaves(DTree))
    plotTree.totalD = float(getTreeDepth(DTree))
    plotTree.xOff = -0.5/plotTree.totalW
    plotTree.yOff = 1.0
    plotTree(DTree, (0.5,1.0), '')
    plt.subplots_adjust(left=0.02,bottom=0.02,right=0.98,top=0.98)
    plt.savefig('./Plot_DTree.png',dpi=200)
    plt.show()

