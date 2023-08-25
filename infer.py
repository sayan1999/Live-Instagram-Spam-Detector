#!/usr/bin/env python
# coding: utf-8



import pandas as pd
import json, glob

import streamlit as st 
from streamlit_echarts import st_echarts


# ## Instaloader api

# In[3]:


from instaloader import Instaloader, Profile, ProfileNotExistsException, LoginRequiredException
L=Instaloader()
import time
from functools import lru_cache 
  
ERROR=''
@lru_cache(maxsize = 100) 
def get_info(username, timeout=0):
    global ERROR
    try:
        time.sleep(timeout)
        print("Scraping ...")
        profile = Profile.from_username(L.context, username=username)
        info = profile._asdict()
        return info
    except ProfileNotExistsException as e:
        print(e)
        ERROR="Profile doesn't exist"
    except LoginRequiredException as e:
        print(e)
        ERROR="Need to login to use instaloader, timeout from instaloader"      


# ### Feature enggineering

# In[4]:


features =[
#     'full_name',
#     'username',
#      'id',
    
    'biography', #len
#  'blocked_by_viewer',
#  'restricted_by_viewer',
 # 'country_block',
#  'external_url',
#  'external_url_linkshimmed',
 'edge_followed_by', #count inside
 'fbid', #is none
#  'followed_by_viewer',
 'edge_follow',  #count inside
#  'follows_viewer',
 
 'has_ar_effects',
 'has_clips',
 'has_guides',
 'has_channel',
    
 # 'has_blocked_viewer',
 'highlight_reel_count',
 # 'has_requested_viewer',
 'is_business_account',
 # 'is_joined_recently',
#  'business_category_name',
#  'overall_category_name',
#  'category_enum',
#  'category_name',
 'is_private',
 'is_verified',
#  'edge_mutual_followed_by',
 'profile_pic_url', #if none
#  'profile_pic_url_hd',
#  'requested_by_viewer',
#  'should_show_category',
 # 'connected_fb_page', #if None or not
#  'edge_felix_video_timeline',
#  'edge_owner_to_timeline_media',
#  'edge_saved_media',
#  'edge_media_collections'
]

def process_entries(ent):
    ent=ent.copy()
    ent['biography']=len(ent['biography'])
    ent['fbid']=bool(ent['fbid'])
    ent['profile_pic_url']=bool(ent['profile_pic_url'])
    
    ent['edge_followed_by']=ent['edge_followed_by']['count']
    ent['edge_follow']=ent['edge_follow']['count']
    
#     ent.pop('full_name'), ent.pop('id'), ent.pop('username')
    return ent


# ## Collect spam users data
# https://www.kaggle.com/datasets/rezaunderfit/instagram-fake-accounts-dataset

# In[5]:




fakedata=[json.load(open(f))['graphql']['user'] for f in glob.glob('dataset/fake/db/*.json')]


# ### Collect real users 
# https://raw.githubusercontent.com/harshitkgupta/Fake-Profile-Detection-using-ML/master/data/users.csv

# In[6]:


import numpy as np

# In[7]:


import os
from tqdm import tqdm

# realusertxt='dataset/real/real.txt'
realjson='dataset/real/real.json'
realcsv='dataset/real/users.csv'

if os.path.isfile(realjson):
    realdata = json.load(open(realjson))
else:
    realdata = []
    print('One time setup to be done, this may take a while...')
    
savedusers = [row['username'] for row in realdata]
    
# for realuser in tqdm(open(realusertxt).readlines()):

        
# json.dump(realdata, open(realjson, 'w+'))


# In[ ]:





# In[8]:


fakedf = pd.DataFrame(map(process_entries, fakedata), columns=features)
fakedf['fake']=1
realdf = pd.DataFrame(map(process_entries, realdata), columns=features)
realdf['fake']=0
df = pd.concat([fakedf, realdf])
df=df.sample(frac=1).reset_index(drop=True)
df.head()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# ## Oversampling imbalanced data

# In[9]:


from imblearn.over_sampling import SMOTE
X, y = SMOTE().fit_resample(df.drop(columns=['fake']), df['fake'])


# In[ ]:





# In[ ]:





# ## Model

# In[10]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate, cross_val_score



# In[ ]:





# In[ ]:





# In[11]:


model=RandomForestClassifier()
model.fit(X, y)


# ## Test on realdata

# In[16]:

@st.cache_data
def predict(username):
    np.random.seed(1111)
    info = get_info(username)
    if info is None:
        return 404
    return model.predict_proba(pd.DataFrame([process_entries(info)], columns=features))[0][1]

# In[ ]:



# Title
st.set_page_config(page_title='CatchSpam',)
st.title("Instagram Spam Profile Checker")



with st.form('input_form'):
    username = st.text_input("Enter Profile Username here", '')
    clickSubmit = st.form_submit_button('Submit')




if clickSubmit:
    predicted = predict(username)
    if predicted != 404:
        option = {
            "series": [
                {
                    "type": "gauge",
                    "startAngle": 180,
                    "endAngle": 0,
                    "min": 0,
                    "max": 100,
                    "center": ["50%", "50%"],
                    "splitNumber": 4,
                    "axisLine": {
                        "lineStyle": {
                            "width": 6,
                            "color": [
                                [0.25, "#64C88A"],
                                [0.5, "#FDDD60"],
                                [0.75, "#ffa500"],
                                [1, "#FF403F"],
                            ],
                        }
                    },
                    "pointer": {
                        "icon": "path://M12.8,0.7l12,40.1H0.7L12.8,0.7z",
                        "length": "12%",
                        "width": 30,
                        "offsetCenter": [0, "-60%"],
                        "itemStyle": {"color": "auto"},
                    },
                    "axisTick": {"length": 10, "lineStyle": {"color": "auto", "width": 2}},
                    "splitLine": {"length": 15, "lineStyle": {"color": "auto", "width": 5}},
                    "axisLabel": {
                        "fontSize": 12,
                        "distance": -60,
                    },
                    "title": {"offsetCenter": [0, "-20%"], "fontSize": 20,  "color": "#FF403F" if predicted > 0.75 else "#ffa500" if predicted > 0.5 else "#FDDD60" if predicted > 0.25 else "#64C88A"},
                    "detail": {
                        "fontSize": 15,
                        "offsetCenter": [0, "0%"],
                        "valueAnimation": True,
                        "color": "auto",
                        "formatter": "Spam Score: {value}%",
                    },
                    "data": [{"value": round(predicted*100), "name": 'Spam' if predicted > 0.75 else 'Suspicious' if predicted > 0.5 else 'Good' if predicted > 0.25 else 'Genuine'}],
                }
            ]
        }    
        st_echarts(option, width="450px", height="350px", key="gauge")
    else:
        st.text(f"Error in processing, please recheck username and retry ...\nError: {ERROR}")


