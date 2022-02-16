import numpy as np
from qpsolvers import solve_qp
import random
def unfairnessDoc(obs,relevance_esti_orig,fairness_strategy,**kwargs):

    """
    return the exposure disparity of each documents according to different disparity strategy.
    """
    relevance_esti_orig=np.clip(relevance_esti_orig,0,np.inf)
    obs=obs
    relevance_esti=relevance_esti_orig     
    doc_num=obs.shape[0]
    if fairness_strategy=="FairCo": 
        ##please refer to https://arxiv.org/abs/2005.14713
        relevance_esti_clip=np.clip(relevance_esti,1e-2,np.inf)  ## to avoid zero.
        ratio = obs/relevance_esti_clip
        swap_reward=ratio[:,None]-ratio[None,:] ## Eq.10 in above paper
        unfairness = np.max(swap_reward,axis=0) ## the equation after Eq.16    

    elif fairness_strategy=="FairCo_maxnorm": 
      relevance_esti_clip=np.clip(relevance_esti,1e-2,np.inf)  ## to avoid zero.
      ratio = obs/relevance_esti_clip
      swap_reward=ratio[:,None]-ratio[None,:]
      unfairness = np.max(swap_reward,axis=0)
      unfairness=unfairness/(np.max(np.abs(unfairness))+1e-10)

    elif fairness_strategy=="FairCo_multip.": 
      swap_reward = obs[:,None]*relevance_esti[None,:]
      unfairness = np.max(swap_reward-swap_reward.T,axis=0)
      unfairness=unfairness/(np.max(np.abs(unfairness))+1e-10)

    elif fairness_strategy=="FairCo_average":
      swap_reward = obs[:,None]*relevance_esti[None,:]
      q_result = np.sum((swap_reward-swap_reward.T)*relevance_esti[:,None],axis=0)/(doc_num*(doc_num-1))
      unfairness=q_result
      unfairness=unfairness/(np.max(np.abs(unfairness))+1e-10)

    elif fairness_strategy=="FFC":
      exposure_quota=kwargs["exposure_quota"]
      exposure_left = exposure_quota-obs
      exposure_feasible_id=exposure_left>0
      relevance_esti_rank= np.copy(relevance_esti)
      relevance_esti_rank[exposure_feasible_id]=relevance_esti[exposure_feasible_id]+10
      unfairness=relevance_esti_rank

    elif fairness_strategy=="FFC_v1":
      exposure_quota=kwargs["exposure_quota"]
      exposure_left = (exposure_quota-obs)
      unfairness=exposure_left

    else:
      raise     
    return unfairness

def multiple_rankings(scores, rankListLength):
    """
    This function gives multiple ranking lists according to the scores, and the length of each ranking list is rankListLength.
    """
    n_samples = scores.shape[0]
    n_docs = scores.shape[1]
    rankListLength = min(n_docs, rankListLength)
    ind = np.arange(n_samples)
    rankingScore=-scores
    partition = np.argpartition(rankingScore, rankListLength, axis=1)[:,:rankListLength]
    sorted_partition = np.argsort(rankingScore[ind[:, None], partition], axis=1)
    rankings = partition[ind[:, None], sorted_partition]
    return rankings
def single_ranking(score, rankListLength):
    """
    This function gives a ranking according to the score, and the length of the ranking list is rankListLength.
    """
    ranking=multiple_rankings(score[None,:],rankListLength)[0]
    return ranking

def updateExposure(qid,dataSplit,ranking,positionBias):
    """
    This function update the exposure.
    """
    qExpVector=dataSplit.query_values_from_vector(qid,dataSplit.exposure)
    qExpVector[ranking]+=positionBias
def getQuotaFromQP(relevance_esti,obs,FutureFairExpo,positionBias):
    """
    This function calculate the additional quota needed for each document via qudratic optimization.
    """
    sum_sqaure_R=np.sum((relevance_esti)**2)
    sum_mutiply_ER=np.sum(relevance_esti*obs)
    B=obs*sum_sqaure_R-relevance_esti*sum_mutiply_ER
    n_docs=relevance_esti.shape[0]
    A=sum_sqaure_R*np.identity(n_docs)-relevance_esti[:,None]*relevance_esti[None,:]+np.identity(n_docs)*1e-4
    lb=np.zeros(n_docs)
    ub=np.ones(n_docs)*FutureFairExpo*positionBias[0]/positionBias.sum()
    Constant_A=np.ones(n_docs)
    x = solve_qp(A, B, A=Constant_A, b=FutureFairExpo,lb=lb,ub=ub)
    if x is None:
        return np.zeros(n_docs).astype(np.float)
    return x
def getQuotaEachItem(obs,q_rel,positionBias,n_futureSession,fairness_tradeoff_param):
    """
    This funciton calcultes the Quota each item should get to keep fairness.
    """
    FutureFairExpo=positionBias.sum()*n_futureSession*fairness_tradeoff_param
    QuotaEachItem=getQuotaFromQP(q_rel,obs,FutureFairExpo,positionBias)
    # print(QuotaEachItem,q_rel/(q_rel.sum()+1e-5)*FutureFairExpo)
    return QuotaEachItem.astype(np.float)

def getExpoBackwardCum(n_futureSession,rankListLength,positionBias):
    """
    This funciton calcultes the Exposure cumulation at each position backwards.
    """
    ExpoBackwardCum=np.zeros((n_futureSession,rankListLength))
    cum=0
    for j in range(rankListLength-1,-1,-1):
        for i in range(n_futureSession-1,-1,-1):
            cum=cum+positionBias[j]
            ExpoBackwardCum[i,j]=cum    
    return ExpoBackwardCum


def getVerticalRanking(q_rel,rankListLength,n_futureSession,ExpoBackwardCum,QuotaEachItem,positionBias):
    """
    This funciton outputs ranklists by an vertical way.
    """
    Expo=np.zeros_like(q_rel)
    rankLists=[]
    n_doc=len(q_rel)
    for i in range(n_futureSession):
        rankLists.append([])
    for j in range(rankListLength):
        for i in range(n_futureSession):
            ranking=rankLists[i]
            q_relCur=q_rel+np.random.uniform(0,0.01,q_rel.shape) #to break tie
            q_relCur[ranking]=-np.inf
            if QuotaEachItem.sum()>=ExpoBackwardCum[i,j]:
                QuotaSatisfiedId=np.where(QuotaEachItem<=0)[0]
                mask=list(set(QuotaSatisfiedId.tolist()+ranking))
                if len(mask)!=n_doc:
                    q_relCur[mask]=-np.inf
            item_ij=np.argmax(q_relCur)
            QuotaEachItem[item_ij]-=positionBias[j]
            QuotaEachItem=np.clip(QuotaEachItem,0,np.inf)
            ranking.append(item_ij)
            Expo[item_ij]+=positionBias[j]
    return rankLists
def getFutureRanking(obs,q_rel,positionBias,n_futureSession,rankListLength,fairness_tradeoff_param):
    """
    This funciton outputs future rankists by conidering fair exposure gurantee.
    """    
    QuotaEachItem=getQuotaEachItem(obs,q_rel,positionBias,n_futureSession,fairness_tradeoff_param)
    ExpoBackwardCum=getExpoBackwardCum(n_futureSession,rankListLength,positionBias)
    rankLists=getVerticalRanking(q_rel,rankListLength,n_futureSession,ExpoBackwardCum,QuotaEachItem,positionBias)
    random.shuffle(rankLists)
    return rankLists
def get_ranking(qid,dataSplit,fairness_strategy,fairness_tradeoff_param,rankListLength,n_futureSession=None,positionBias=None):
    """
    This function return the ranking.
    """
    qRel=dataSplit.query_values_from_vector(qid,dataSplit.label_vector)
    qExpVector=dataSplit.query_values_from_vector(qid,dataSplit.exposure)
    if fairness_strategy in ["FairCo","FairCo_maxnorm",'FairCo_multip.',"FairCo_average"]:
      Docunfairness=unfairnessDoc(qExpVector,qRel,fairness_strategy)
      RankingScore=qRel+fairness_tradeoff_param*Docunfairness
      ranking=single_ranking(RankingScore,rankListLength=rankListLength)
    elif fairness_strategy in ["QPfair"]:
      if len(dataSplit.cacheLists[qid])<=0:
        dataSplit.cacheLists[qid]=getFutureRanking(qExpVector,qRel,positionBias,n_futureSession,rankListLength,fairness_tradeoff_param)
      ranking=np.array(dataSplit.cacheLists[qid].pop())
    elif fairness_strategy == "Randomk":
      RankingScore=np.random.uniform(0,1,qRel.shape)
      ranking=single_ranking(RankingScore,rankListLength=rankListLength)     
    elif fairness_strategy == "Topk":
      RankingScore=qRel
      ranking=single_ranking(RankingScore,rankListLength=rankListLength)    
    return ranking