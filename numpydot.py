"""
@author: gurkankilicaslan
Hello!!!
This code does everything NumPy dot function does.
It gives the exact same result with numpy.dot

I have added some variables that you might wanna try for the numpydot function 
that I created.

numpydot function doesn't allow unreasonable pairs as numpy.dot. 
It basically has the features of numpy.dot.
I recommend you to check the results comparing with numpy.dot.

You can find the numpy.dot utilities from the following link:
    https://www.sharpsightlabs.com/blog/numpy-dot/

"""
# # These are some variables:
# a = 2
# b = 3

# c = [4]
# d = [5]

# e = [2, 4, 3]
# f = [4, 5, 6]


# g = [[1,2,3],[4,5,6]]
# h = [[1,2,3],[4,5,6],[7,8,9]]
# i = [[1,2,3,4],[4,5,6,7],[7,8,9,10]]

def numpydot(x1, x2):
    
    if type(x1) != list and type(x2) != list:
        ret = x1*x2
        
    elif type(x1) != list and type(x2) == list and type(x2[0]) != list:
        outer = []
        for i in range(len(x2)):
            outer.append(x1*x2[i])
        ret = outer
    elif type(x2) != list and type(x1) == list and type(x1[0]) != list:
        outer = []
        for i in range(len(x1)):
            outer.append(x2*x1[i])
        ret = outer
        
        
    elif type(x1) != list and type(x2) == list and type(x2[0]) == list:
        outer = []
        for i in range(len(x2)):
            inner = []
            for j in range(len(x2[0])):
               inner.append(x1*x2[i][j])
            outer.append(inner)
        ret = outer
    elif type(x2) != list and type(x1) == list and type(x1[0]) == list:
        outer = []
        for i in range(len(x1)):
            inner = []
            for j in range(len(x1[0])):
               inner.append(x2*x1[i][j])
            outer.append(inner)
        ret = outer
    
    
    elif type(x1) == list and type(x2) == list and type(x1[0])!=list and type(x2[0]) != list and len(x1) == len(x2):
        ret = sum(i*j for i, j in zip(x1, x2))  
        
        
    elif type(x1[0]) != list and type(x2[0]) == list and len(x1)==len(x2):
        outer = []
        for i in range(len(x2[0])): 
            total = 0
            for j in range(len(x1)): 
                total += x1[j] * x2[j][i]
            outer.append(total)
            ret = outer
            
    elif type(x2[0]) != list and type(x1[0]) == list and len(x2)==len(x1[0]):
        outer = []
        for i in range(len(x1)): 
            total = 0
            for j in range(len(x2)): 
                total += x2[j] * x1[i][j]
            outer.append(total)
            ret = outer
            
        
    elif type(x1[0]) == list and type(x2[0]) == list and  len(x1[0]) == len(x2):
        outer = []
        for i in range(len(x1)): 
            inner = []
            for j in range(len(x2[0])): 
                total = 0
                for k in range(len(x1[0])):
                    total += x1[i][k] * x2[k][j]
                inner.append(total)
            outer.append(inner)
            ret = outer
    
    return ret


# print(numpydot(a,i),"\n")






