import matplotlib.pyplot as plt 
desiredGradFair=["Topk","ExploreK","FairCo","ILP","LP","FairK(Ours)","GradFair(Ours)"]
def reorder(desiredList,curList):
    """
    This function index mapping from the curList according to desiredList.
    """
    reoderIndex=[]
    for legend in desiredList:
        reoderIndex.append(curList.index(legend))
    return reoderIndex
def reorderLegend(desiredLegend,ax):
    """
    This function index reorder the legend to desiredLegend.
    """
    handles, labels = plt.gca().get_legend_handles_labels()
    order=reorder(desiredLegend,labels)
    ax.legend([handles[idx] for idx in order],[labels[idx] for idx in order])




