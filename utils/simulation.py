import numpy as np
def sample_queryFromdata(data,query_rng):
    """
    This funciton samples a query from all three splits, data.train, data.validation, and data.test.
    """
    dataSplits=[data.train,data.validation,data.test]
    splitId=sample_splitId(dataSplits,query_rng)
    Queryid=sample_Queryid(dataSplits[splitId],query_rng)
    dataSplits[splitId].query_freq[Queryid]+=1
    return Queryid,dataSplits[splitId]

def sample_splitId(dataSplits,query_rng):
    """
    This funciton samples a data split from data.train, data.validation, and data.test.
    """
    n_queries =[len(dataSplit.queriesList) for dataSplit in dataSplits]
    total_n_query=sum(n_queries)
    query_ratio = np.array(n_queries)/total_n_query

    splitId = query_rng.choice(3, size=1,
                                     p=query_ratio)
    return  int(splitId)
def sample_Queryid(dataSplit,query_rng):
    """
    This funciton samples a data split from data.train, data.validation, and data.test.
    """
    Queryid = query_rng.choice(dataSplit.queriesList, size=1)[0]
    return  Queryid
def getpositionBias(cutoff,positionBiasSeverity):
    return (1/np.log2(2+np.arange(cutoff)))**positionBiasSeverity
