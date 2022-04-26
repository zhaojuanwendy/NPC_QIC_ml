#!/usr/bin/env python
# coding: utf-8

# In[62]:


import numpy as np
import sklearn.datasets as dt
from sklearn.mixture import GaussianMixture


# In[63]:


seed = 11
rand = np.random.RandomState(seed)


# In[64]:


mu_1=20
sigma_1=5
score_set1 = np.random.normal(mu_1, sigma_1, 1000)


# In[65]:


mu_2=100
sigma_2=5
score_set2 = np.random.normal(mu_2, sigma_2, 1000)


# In[66]:


combined_score = np.concatenate((score_set1, score_set2))
combined_score.shape


# In[67]:


combined_score = np.expand_dims(combined_score, 1)
combined_score.shape


# In[68]:


gmm_init = np.expand_dims(
    [np.min(combined_score), np.max(combined_score)], 1)


# In[73]:


gmm_noconv = GaussianMixture(n_components=2, means_init=gmm_init, max_iter=1)

# Fitting Gaussian mixture model and determining posterior probablity
gmm_noconv.fit(combined_score)
y_test_not_converged = gmm_noconv.predict_proba(combined_score)[:, 1]
y_test_not_converged


# In[79]:


gmm_cov = GaussianMixture(n_components=2, means_init=gmm_init, max_iter=100)
print("if converged", gmm_cov.precisions_)
# Fitting Gaussian mixture model and determining posterior probablity
gmm_cov.fit(combined_score)
y_test_converged = gmm_cov.predict_proba(combined_score)[:, 1]


# In[75]:


y_test_converged


# In[76]:


np.array_equal(y_test_not_converged, y_test_converged)


# In[72]:
#
#
# import matplotlib.pyplot as plt
#
# # Making plot
# x = np.linspace(np.min(combined_score), np.max(
#     combined_score), len(combined_score))
# y = np.exp(gmm.score_samples(x.reshape(-1, 1)))
#
#
# fig = plt.figure(figsize=[14, 6])
# ax1 = plt.subplot(121)
# ax1.hist(combined_score, color=['dimgray'], density=True, bins=20)
# ax1.plot(x, y, lw=4, label="GMM")
# plt.xlabel("PheMap Phenotype Score", fontsize=16)
# plt.ylabel("Density", fontsize=16)
# plt.title('Gaussian Mixture Model Fit to All Phescores', fontsize=20)
# plt.show


# In[ ]:




