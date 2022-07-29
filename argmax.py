def argmax(vector):
    index, value = 0, vector[0]
    for i,v in enumerate(vector):
        if v > value:
            index, value = i,v
    return index
vector = list(input())
result = argmax(vector)
print('arg max of %s: %d' % (vector, result))
