#global 
import abc 
import numpy as ivy 
#creating arry from array
#function 

#ToDo implement all methods of linear algebra 

Class ArrayWithLinalg(abc.ABC):
  abc.ABC = ABC
  
  def getABC(abc):
    retun abc.ABC
    
    #vecdot 
    #values are taken for code purpose
    vector_A = ivy.array([[2+3j]])
    vector_B = ivy.array([[3+4j]])
    vector_C = ivy.array([[4+5j]])
    
    abc = ivy.dot(vector_A, vector_B, vector_C)
    print("Dot Product", abc)
    
    # for n-dimentional array or tensores it is sum of product over the last axis 
    # like ivy.dot(a,b,c)[i,j,k,l,m,n] = sum(a[i,j, :] * b[k,l, :] * c[m,n, :]
