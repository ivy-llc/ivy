import ivy
ivy.set_backend('paddle')

ar1= ivy.array(0)
ar2= ivy.array(0)
assume_unique=False

print("===============================================")
print("\n")
print(f"ar1: {ar1} | dtype(ar1): {ivy.dtype(ar1)}")
print(f"ar2: {ar2} | dtype(ar2): {ivy.dtype(ar2)}")
print(f"assume_unique: {assume_unique}")

common_dtype = ivy.promote_types(ivy.dtype(ar1), ivy.dtype(ar2))
ar1 = ivy.asarray(ar1, dtype=common_dtype)
ar2 = ivy.asarray(ar2, dtype=common_dtype)
print("after common dtype")
print(f"ar1: {ar1}")
print(f"ar2: {ar1}")

if not assume_unique:
    ar1 = ivy.unique_values(ar1)
    ar2 = ivy.unique_values(ar2)
print(f"ar1: {ar1} shape: {ivy.shape(ar1)} dtpye: {ivy.dtype(ar1)}")
print(f"ar2: {ar1} shape: {ivy.shape(ar2)} dtpye: {ivy.dtype(ar2)}")

ar1 = ivy.reshape(ar1, (-1,))
ar2 = ivy.reshape(ar2, (-1,))
print("after reshaping to -1")
print(f"ar1: {ar1}")
print(f"ar2: {ar1}")

print("after unique values selecting")
print(f"ar1: {ar1}")
print(f"ar2: {ar1}")

aux = ivy.concat([ar1, ar2], axis=0)
print(f"aux: {aux}")

if aux.size == 0:
    print("aux is empty!")
    print(aux)

aux = ivy.sort(aux)
flag = ivy.concat((ivy.array([True]), ivy.not_equal(aux[1:], aux[:-1]), ivy.array([True])), axis=0)
print(f"flag: {flag}")
mask = flag[1:]&flag[:-1]
if ivy.all(ivy.logical_not(mask)):
    ret = ivy.asarray([])
else:
    ret = aux[mask]
print(f"aux[flag]: {ret}")


# ivy.array([False,False]