import numpy as np
import pandas as pd

dict = {}


'''
raw_data = "Node.txt"
dataset = np.loadtxt(raw_data, delimiter=",")
key = dataset[:,0]
value = dataset[:,2]
l2 = list(set(key)) #去重后的key
key = list(key)
value = list(value)
for i in range(len(l2)):
    l4 = []
    fea = []
    for j in range(len(key)-1,-1,-1):
        if l2[i] == key[j]:
            l4. append(value[j])
            key.pop(j)
            value.pop(j)
        else:
            continue;
    print(len(l4))
    l4 = np.array(l4,dtype = 'float_')
    mean = np.mean(l4)
    std  = np.std(l4,ddof = 1)
    min  = np.min(l4)
    max  = np.max(l4)
    l4.sort()
    q = np.percentile(l4,[25, 50, 75, 100])
    fea.append(mean)
    fea.append(std)
    fea.append(min)
    fea.append(max)
    fea.append(q[0])
    fea.append(q[1])
    fea.append(q[2])
    fea.append(q[3])
    dict.setdefault(l2[i],fea)
'''

file = open("Node.txt")
l1 = [] #key
l5 = [] #存所有节点的数据
#先存数据，l1只存key(即为变量值),l5存变量和value值
while 1:
    lines = file.readlines(100000)
    if not lines:
        break
    for line in lines:
        l0 = [] #key,value
        key = line.split(',')[0]
        objincrease = line.split(',')[2]
#        a = line.split(',')[4]
#        b = line.split(',')[5]
#        sbscore = line.split(',')[3]
        l1.append(key)
        l0.append(key)
        l0.append(objincrease)
#        l0.append(a)
#        l0.append(b)
#        l0.append(sbscore)
        l5.append(l0)
            

l2 = list(set(l1)) #去重后的key

#从l5中找到与l2中每个相匹配的键对应的
for i in range(len(l2)):
    l4 = []#存储每个键对应的值
    fea = []
    for j in range(len(l5)-1,-1,-1):
        if l2[i] == l5[j][0]:
            l4.append(l5[j][1])
            l5.pop(j)
        else:
            continue;
    print(len(l4)) 
    l4 = np.array(l4,dtype = 'float_')
    mean = np.mean(l4)
    std  = np.std(l4,ddof = 1)
    min  = np.min(l4)
    max  = np.max(l4)
    l4.sort()
    q = np.percentile(l4,[25, 50, 75, 100])  
    fea.append(mean)
    fea.append(std)
    fea.append(min)
    fea.append(max)
    fea.append(len(l4)/4143)
    fea.append(q[0])
    fea.append(q[1])
    fea.append(q[2])
    fea.append(q[3])
    dict.setdefault(l2[i],fea)
print()
key = dict.keys()
print(key)
print()
print(len(l2))
print (len(dict))
for i in range(len(l2)):
    key = l2[i]
    values = dict.get(key)
    print(key,len(values),values)
    
