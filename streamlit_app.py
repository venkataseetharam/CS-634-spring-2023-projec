#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
from transformers import pipeline


# In[2]:


def analyze_sentiment(model_name, text):
    classifier = pipeline('sentiment-analysis', model=model_name)
    result = classifier(text)
    label = result[0]['label']
    

    if label == 'LABEL_1':
        d='Negative'
        st.error(f'{d}')
    else:
        d='Positive'
        st.success(f'{d}')
        


# In[3]:


def main():
    st.title('Sentiment Analysis App')
    
    # Get user input
    text = st.text_area('Enter text')
    model_name = st.selectbox('Select a model', ['distilbert-base-uncased', 'bert-base-uncased', 'roberta-base'])

    # Analyze sentiment
    if st.button('Submit'):
        sentiment = analyze_sentiment(model_name, text)
       
        
if __name__ == '__main__':
    main()


# In[ ]:



