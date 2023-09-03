import ivy 

a = ivy.array([1,2,3],dtype="float32")
print(a.dtype == "float32")
print(isinstance(a, ivy.data_classes.array.array.Array))