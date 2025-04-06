#!/usr/bin/env python
# coding: utf-8

# In[61]:


from fastai.vision.all import *


# In[62]:


# 安装 gradio 库
get_ipython().system('pip install gradio')


# In[63]:


import gradio as gr


# In[64]:


import warnings

# 忽略fastai的pickle警告
warnings.filterwarnings("ignore", category=UserWarning, module="fastai.learner")


# In[65]:


import os


# In[66]:


import shutil


# In[67]:


# 定义与训练时相同的函数
def is_cat(x): return x[0].isupper()


# In[68]:


# 确保模型文件存在
os.makedirs('model', exist_ok=True)
source_path = '/Users/jingjingzhao/.fastai/data/oxford-iiit-pet/images/models/model1.pkl'
target_path = 'model/model1.pkl'


# In[69]:


# 如果目标路径不存在，则复制模型文件
if not os.path.exists(target_path) and os.path.exists(source_path):
    shutil.copy(source_path, target_path)


# In[70]:


def predict_image(img):
    try:
        # 优先尝试使用相对路径
        if os.path.exists(target_path):
            learn = load_learner(target_path)
        else:
            # 如果相对路径不存在，尝试使用原始绝对路径
            learn = load_learner(source_path)
            
        # 进行预测
        pred_class, pred_idx, probs = learn.predict(img)
        return {
            "Prediction": "Cat" if pred_class else "Dog",
            "Confidence": f"{probs[pred_idx]:.4f}"
        }
    except Exception as e:
        return {"Error": str(e)}


# In[71]:


# 创建Gradio界面
interface = gr.Interface(
    fn=predict_image,
    inputs=gr.Image(),
    outputs=gr.JSON(),
    title="Cat or Dog?",
    description="Uploading a picture, I will recognize if it is cat or dog"
)


# In[72]:


interface.launch()


# In[ ]:




