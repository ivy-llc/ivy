#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import ivy
def argmax(value,axis,name="argmax"):
    return ivy.argmax(value,axis)
argmax.unsupported_dtypes={"torch":("float16", "bfloat16")}

