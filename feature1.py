import math
import numpy as np
import cplex as CPX
maxinfinity = +CPX.infinity
mininfinity = -CPX.infinity

#调整读取模式
def p1(listf):
    if len(listf) > 0 :
         a1 = np.sign(min(listf))
         a2 = np.sign(max(listf))
         a3 = np.fabs(min(listf))
         a4 = np.fabs(max(listf))
    else:
         a1 = 0
         a2 = 0
         a3 = 0
         a4 = 0
    return a1,a2,a3,a4;
#inf = maxinfinity , -inf = mininfinity , nan = 0
def floatchange(a):
    if a == np.inf or np.isnan(a):
        a = maxinfinity
    elif a == -np.inf:
        a = mininfinity
    else: 
        a = a
    return a 


#i代表行，j代表列
def f1(listc,lista,listb):
    f1 = []
    #只用计算一次即可
    sum1 = np.sum(np.maximum(listc,0))
    sum2 = np.sum(np.minimum(listc,0))
    #j代表列，即代表每个变量
    for j in range(len(lista[1])):
        list = []
        one = np.sign(listc[j])
        two = np.true_divide(listc[j],np.fabs(sum1))
        three = np.true_divide(listc[j],np.fabs(sum2))
        two = floatchange(two)
        three = floatchange(three) 
#        list.append(vars[j])
        list.append(one)
        list.append(two)
        list.append(three)
        list1 = []
        list2 = []
        list3 = []
        list4 = []
        list5 = []
        list6 = []
        list7 = []
        list8 = []
        #i代表行，即代表每个约束
        for i in range(len(lista)):
            if listb[i] >= 0:#可以用到numpy的数值计算
                aa = np.true_divide(lista[i][j],np.fabs(listb[i]))
                aa = floatchange(aa) 
                list1.append(aa)
            else:
                bb = np.true_divide(lista[i][j],np.fabs(listb[i]))
                bb = floatchange(bb)
                list2.append(bb)
            if listc[j] > 0:
                aa = np.true_divide(np.fabs(listc[j]),lista[i][j])
                aa = floatchange(aa)
                list3.append(aa)
            else:
                bb = np.true_divide(np.fabs(listc[j]),lista[i][j])
                bb = floatchange(bb)
                list4.append(bb)

            #有多少行就要计算多少次，但对于每个lista[i][j],他们只用计算一次
            sum11 = np.sum(np.maximum(lista[i],0))
            sum22 = np.sum(np.minimum(lista[i],0))
            if lista[i][j] >= 0:
                aa = np.true_divide(lista[i][j],sum11)
                aa = floatchange(aa)
                bb = np.true_divide(lista[i][j],sum22)
                bb = floatchange(bb)
                list5.append(aa)
                list6.append(bb)
            else:
                aa = np.true_divide(lista[i][j],sum11)
                aa = floatchange(aa)
                bb = np.true_divide(lista[i][j],sum22)
                bb = floatchange(bb)
                list7.append(aa)
                list8.append(bb)

        pa = p1(list1)
        p2 = p1(list2)
        p3 = p1(list3)
        p4 = p1(list4)
        p5 = p1(list5)
        p6 = p1(list6)
        p7 = p1(list7)
        p8 = p1(list8)
        list.append(pa[0]) 
        list.append(pa[1]) 
        list.append(pa[2]) 
        list.append(pa[3])

        list.append(p2[0]) 
        list.append(p2[1]) 
        list.append(p2[2]) 
        list.append(p2[3])

        list.append(p3[0]) 
        list.append(p3[1]) 
        list.append(p3[2]) 
        list.append(p3[3])

        list.append(p4[0]) 
        list.append(p4[1]) 
        list.append(p4[2]) 
        list.append(p4[3])

        list.append(p5[0]) 
        list.append(p5[1]) 
        list.append(p5[2]) 
        list.append(p5[3])

        list.append(p6[0]) 
        list.append(p6[1]) 
        list.append(p6[2]) 
        list.append(p6[3])

        list.append(p7[0]) 
        list.append(p7[1]) 
        list.append(p7[2]) 
        list.append(p7[3])

        list.append(p8[0]) 
        list.append(p8[1]) 
        list.append(p8[2]) 
        list.append(p8[3])
        f1.append(list)
    return f1;
    

