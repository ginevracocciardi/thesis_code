
# coding: utf-8

# In[1]:


from sklearn.metrics import log_loss


# In[2]:


def calculate_log_loss(class_ratio,multi=10000):
    
    if sum(class_ratio)!=1.0:
        print("warning: Sum of ratios should be 1 for best results")
        class_ratio[-1]+=1-sum(class_ratio)  
    
    actuals=[]
    for i,val in enumerate(class_ratio):
        actuals=actuals+[i for x in range(int(val*multi))]
        

    preds=[]
    for i in range(multi):
        preds+=[class_ratio]

    return (log_loss(actuals, preds))


# In[3]:


logloss=calculate_log_loss([.16,.3,.54],multi=100)
logloss

