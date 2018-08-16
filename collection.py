#!/usr/bin/python
# ---------------------------------------------------------------------------
# File: admipex1.py
# Version 12.8.0
# ---------------------------------------------------------------------------
# Licensed Materials - Property of IBM
# 5725-A06 5725-A29 5724-Y48 5724-Y49 5724-Y54 5724-Y55 5655-Y21
# Copyright IBM Corporation 2009, 2017. All Rights Reserved.
#
# US Government Users Restricted Rights - Use, duplication or
# disclosure restricted by GSA ADP Schedule Contract with
# IBM Corp.
# ---------------------------------------------------------------------------
"""
Use the node and branch callbacks for optimizing a MIP problem.

To run this example, the user must specify a problem file.

You can run this example at the command line by

    python admipex1.py <filename>
"""

from __future__ import print_function
from sklearn.linear_model import LinearRegression
from math import floor, fabs,log,ceil,log
from functools import reduce


import cplex as CPX
from cplex.callbacks import BranchCallback,NodeCallback,SolveCallback,IncumbentCallback
from cplex._internal._subinterfaces import SensitivityInterface,BasisInterface,AdvancedSolutionInterface,SolutionInterface,VariablesInterface,LinearConstraintInterface,ObjectiveInterface


import sys 
import numpy as np
from userpython import feature1
from random import choice
import time
#用来存储收集到的数据集
doc = open("/root/pro/data/BP102_12.txt",'a')
filename = "/root/pro/project/cplexpython/lp/BPEQ_102.lp"
c1 = CPX.Cplex(filename)
lc = LinearConstraintInterface(c1)
v = VariablesInterface(c1)
o = ObjectiveInterface(c1)
s = SolutionInterface(c1)


nint = c1.variables.get_num_integer()
nbin = c1.variables.get_num_binary()
nvar = c1.variables.get_num()
vars = c1.variables.get_names()
cons = c1.linear_constraints.get_names()

#创建一个全局变量，以字典形式存储其中产生的每个nodeid和其对应的value
nodeval = {} #用来存储NodeCallback中产生的所有节点数据
nodeval1 = {} #用来存储BranchCallback中产生的所有节点数据----------可以将BranchCallback中的数据传给NodeCallback使用，也可以互传
#创建一个全局变量，以字典形式存储其中产生的每个nodeid和其对应的branchvar
nodevar = {}
nodevar1 = {}
nodevarval = {}
setbranchvar = set()
#创建一个全局变量，以字典形式存储其中产生的每个nodeid和其对应的objinc
nodeoi = {}
#创建一个全局变量，以字典形式存储，统计每个变量观察到的目标增长
objinc = {}
#创建一个全局变量，含义是总分支次数
totalbranchnum = 0
#创建一个nvars的列表,用来存储针对每个变量观察到的目标增长
listobjinc = [[] for i in range(nvar)]
listobjinc1 = [[] for i in range(nvar)]
listobjinc2 = [[] for i in range(nvar)]
#创建一个全局变量，代表了当前解
varvalues = []


#用来获取灵敏度分析和diebeek_penalty
clone = CPX.Cplex(c1)
clone.set_problem_type(clone.problem_type.LP)
clone.solve()

basic = clone.solution.basis.get_basis()
b = [i for i,v  in enumerate(basic[0]) if v == 1 ] 

#针对变量的
relx = clone.solution.get_values()
dp = clone.solution.advanced.get_Driebeek_penalties(b)
objsa = clone.solution.sensitivity.objective()
#都是针对约束行的
rhssa = clone.solution.sensitivity.rhs()
slacks = clone.solution.get_linear_slacks()
duals = clone.solution.get_dual_values()

#用来获取c,a,b,从而计算第一部分特征
listc = o.get_linear() 

m = lc.get_num()
n = v.get_num()
ma = np.zeros([m,n])
for i in range(m):
    for j in range(n):
        ma[i][j] = lc.get_coefficients(i,j)


lista = list(ma)
  
listb = c1.linear_constraints.get_rhs()
listv = c1.variables.get_names()
#计算第一部分特征
start = time.clock()
f1 = feature1.f1(listc,lista,listb)
end = time.clock()
print(end - start,"ms")

#print("f1 = ",len(f1),f1)

maxinfinity = +CPX.infinity
mininfinity = -CPX.infinity

#inf = maxinfinity , -inf = mininfinity , nan = maxinfinity,为了方便机器学习训练，提前处理好数据集
def floatchange(a):
    if a == np.inf or np.isnan(a):
        a = maxinfinity
    elif a == -np.inf:
        a = mininfinity
    else:
        a = a 
    return a 

#用来计算第二部分特征
#用来计算特征47和48、52和53
def sensitivity_analysis(c,sen):
    list1 = []
    for i in range(len(c)):
        list2 = []
        a1 = np.sign(sen[i][0])
        a2 = np.sign(c[i])
        a3 = np.sign(sen[i][1])
        a4 = np.log(np.true_divide((c[i] - sen[i][0]),fabs(c[i])))
        a5 = np.log(np.true_divide((sen[i][1] - c[i]),fabs(c[i])))
        a1 = floatchange(a1)
        a2 = floatchange(a2)
        a3 = floatchange(a3)
        a4 = floatchange(a4)
        a5 = floatchange(a5)
        list2.append(a1)
        list2.append(a2)
        list2.append(a3)
        list2.append(a4)
        list2.append(a5)
        list1.append(list2)
    return list1;


#根据direbeek penalty计算38-43号特征,针对变量i的
def driebeek_penalty(dp1,o):
    list1 = []
    for i in range(len(dp1)):
        list2 = []
        a1 = np.true_divide(np.log(dp1[i][0]),o)
        a2 = np.true_divide(np.log(dp1[i][1]),o)
        a3 = np.true_divide(np.log(dp1[i][0] + dp1[i][1]),o)
        a4 = np.true_divide(dp1[i][0],o)
        a5 = np.true_divide(dp1[i][1],o)
        a6 = np.true_divide((dp1[i][0] + dp1[i][1]),o)
        a1 = floatchange(a1)
        a2 = floatchange(a2)
        a3 = floatchange(a3)
        a4 = floatchange(a4)
        a5 = floatchange(a5)
        a6 = floatchange(a6)
        list2.append(a1)
        list2.append(a2)
        list2.append(a3)
        list2.append(a4)
        list2.append(a5)
        list2.append(a6)
        list1.append(list2)
    return list1;

objective = 1

def nplog(a,o):
    a1 = np.true_divide(np.log(a),o)
    a1 = floatchange(a1)
    return a1;

#最终得出的第二部分特征
listobjsa = sensitivity_analysis(listc,objsa)
listrhssa = sensitivity_analysis(listb,rhssa)
listdp = driebeek_penalty(dp,objective)


#用来计算第三部分特征,l中存的是针对变量i观察到的目标增长
def feature3(l):
    fea = []
    l4 = np.array(l,dtype = 'float_')
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
    return fea;
#a代表左节点的目标增长，b代表右节点的目标增长
def score(a,b):
    e = 10**-6
    return max(a,e)*max(b,e)


all = []
print("nint1 = ",nint)
print("nbin1 = ",nbin)
print("nvar1 = ",nvar)
print("lenvar1 = ",len(vars))
print("lencons1 = ",len(cons))


class MySolve(SolveCallback):

    def __call__(self):
        self.times_called += 1
        if self.get_num_nodes() < 1:
            self.solve(self.method.primal)
        else:
            self.solve(self.method.dual)
        status = self.get_cplex_status()
        self.use_solution()
#在外部构造了一个方法
def one(l):
    return min(l)
#构造了一个外部类,通过多继承，发现可以调用这个类里面的方法
class NodeCallback1():
      def objsa(self):
          return "hello world";






class MyBranch(BranchCallback, NodeCallback1,NodeCallback, SensitivityInterface,BasisInterface,AdvancedSolutionInterface,SolutionInterface):


    def __call__(self):
        self.times_called += 1
        print("0branch----------------------------------------------------------------------------------------------------------")
        nodedata = self.get_node_data()
        nodeid = self.get_node_ID()
        #获取当前节点中的所有数据 
        bestobjval = self.get_best_objective_value()
        incumbentobjval = self.get_incumbent_objective_value()
        #目前节点的解
        objval = self.get_objective_value()
        x = self.get_values()
         

        #测试方法
        '''
        cpx = self.cpx
        #self1,self2,self3读入的问题都是空的问题，所以无效
        self1 = CPX.Cplex(nodedata)
        self2 = CPX.Cplex(nodeid)
        self3 = CPX.Cplex(self)
        self4 = CPX.Cplex(cpx)
#        self4.solve()
#        print("self1 relax = ",self4.get_values())

        self1.set_problem_type(self1.problem_type.LP)
        self1.solve()
        relx = self1.solution.get_values()
        slacks = self1.solution.get_linear_slacks()
        duals = self1.solution.get_dual_values()
        basic = self1.solution.basis.get_basis()
        b = [i for i,v  in enumerate(basic[0]) if v == 1 ]


        #针对变量的
        dp = self1.solution.advanced.get_Driebeek_penalties(b)
        print("relx = ",len(relx),relx)
#        objsa = self1.solution.sensitivity.objective(1)
        #都是针对约束行的
#        rhssa = self1.solution.sensitivity.rhs()

        print("self1 = ",self1)
#        cpx.set_problem_type(cpx.problem_type.LP)
        print("cpx = ",cpx)

        sets = self.cpx.SOS.get_sets()
#        self.cpx.set_problem_type(self.cpx.problem_type.LP)
#        self.cpx.solve()
#        relx = self.cpx.solution.get_values()
#        objsa = self.cpx.solution.sensitivity.objective()
#        print("objsa = ",len(objsa), objsa)
#        self.cpx.solution.sensitivity.objective()
        print("sets = ",len(sets),sets)
        objsa = self.objsa()
        print("测试方法的可用性 = ",objsa)
        #测试方法的
        '''

        pseudocosts = self.get_pseudo_costs()
        linearslacks = self.get_linear_slacks()
        lowerbounds = self.get_lower_bounds()
        upperbounds = self.get_upper_bounds()
        nodedata = self.get_node_data()
        branchnums = self.get_num_branches()
        #hasincumbet = 1，有解，hasincumbent = 0，无解
        hasincumbent = self.has_incumbent()
        '''
        while hasincumbent is 1:
            incumbentvalues = self.get_incumbent_values()
            fractionalvar = [i for i,v  in enumerate(incumbentvalues) if not v.is_integer() ]
            print("fractionalvarbc = ",len(fractionalvar),fractionalvar)   
            return 
        '''
        #存储当前所有变量的状态，0表示可行，1表示不可行 
        feas = self.get_feasibilities()
        #判断当前节点是否是整数可行的
        boolean = self.is_integer_feasible()

        #判断当前问题的类型
        protype = "LP"
        if boolean == True:
            protype= "MIP"
        else:
            protype= "LP"
 
        #变量不可行状态(1)-----对应于basic
        status1 = self.feasibility_status.infeasible
        #变量可行状态(0)-------对应于at_lower_bound
        status2 = self.feasibility_status.feasible
        #收集所有的候选分支变量
        candiatevar = [i for i,v  in enumerate(feas) if v == 1 ]
        print("candiatevar = ",protype,len(candiatevar),candiatevar,status1,status2)

        #可以对candiatevar进行Driebeek penalties
        #可以对candiatevar进行目标函数系数灵敏度分析
        #可以对candiatevar对应约束行进行右边侧系数灵敏度分析
        #可以对candiatevar进行下界灵敏度分析
        #可以对candiatevar进行上界灵敏度分析
        #可以对candiatevar求上下伪成本
        #可以对candiatevar求下界
        #可以对candiatevar求上界
         
        maxobj = +CPX.infinity
        maxinf = -CPX.infinity
        #用来获取索引r的
        if 0 in linearslacks:
            r = linearslacks.index(0)
        else:
            r = 0

        #在branchcallback中求解score的方式，通过（分之前的最优解-分支后的最优解）/当前问题的解
        i = 1
        while i>=1 and i < branchnums:
            branchvar = self.get_branch(i)
            #当前分支变量的索引
            var1 = branchvar[1][0][0]
            nodevar1[nodeid] = var1
            i = i + 1

        nodes = self.get_num_nodes()
        remaining_nodes = self.get_num_remaining_nodes()
        print()
        lnodeid = 2*nodeid + 1
        rnodeid = 2*nodeid + 2
        print("BB--左孩子节点，右孩子节点 = ",lnodeid in nodeval,rnodeid in nodeval)
        print("nodeid = ",nodeid,type(nodeid),objval)
        #单独用来存储BC中产生的节点
        nodeval1[nodeid] = objval
#        print("listobjinc---bb",len(listobjinc1),listobjinc1)
#        print("nodeval1 = ",len(nodeval1),nodeval1)
#        nodeval[int(nodeid)] = objval
        print("b,i,o = ",bestobjval,incumbentobjval,objval)
#        print("objinc bb = ",len(nodeval),nodeval)
#        print()
#        print("Nnodes,Rnodes = ",nodes,remaining_nodes)




listobj1 = []
class MyNode(NodeCallback,NodeCallback1):
    def __call__(self):
        self.times_called += 1
        print("1Node--------------------------------------------------------------------------------------------------------------")
        nodes = self.get_num_nodes()
        remaining_nodes = self.get_num_remaining_nodes()
        print("Nnodes,Rnodes = ",nodes,remaining_nodes)
        bestobjvalue = self.get_best_objective_value()
        listobj1.append(bestobjvalue)
        incumbentobjvalue = self.get_incumbent_objective_value()
        hasincumbent = self.has_incumbent()
        print("bestobj,incumbentobj,hasincumbent= ",bestobjvalue,incumbentobjvalue,hasincumbent)

        #测试方法的 
        self1 = CPX.Cplex(self)
        print("NC中的self = ",self1)
        #测试方法的
        ''' 
        while hasincumbent == 1:
            incumbentvalues = self.get_incumbent_values()
            fractionalvar = [i for i,v  in enumerate(incumbentvalues) if not v.is_integer() ]
            return
        '''
         
             
        for node in range(self.get_num_remaining_nodes()):
            nodeid = self.get_node_ID(node)
            listall = []
 
            #因为当前nodeid不在原来集合里面，我才存到listobjinc2中去
            if nodeid[0] not in nodeval:
                print("该节点不在nodeval中")
                branchvar = self.get_branch_variable(nodeid)
                objvalue = self.get_objective_value(nodeid)
                #用来来存储对应的目标值
                nodeval[nodeid[0]] = objvalue
                #用来存储对应的的变量          
                nodevar[nodeid[0]] = branchvar
                pnodeid = int(np.ceil(nodeid[0]/2)) - 1 
                lnodeid = 2*pnodeid + 1 
                rnodeid = 2*pnodeid + 2 
                nodenumber = self.get_node_number(nodeid)
                #求得当前节点的深度
                depth = self.get_depth(nodeid)
                #求得父节点的深度
                pdepth = depth - 1
                nodedata = self.get_node_data(nodeid)

                if lnodeid in nodeval:
                    lnodeobj = nodeval.get(lnodeid)
                elif lnodeid in nodeval1:
                    lnodeobj = nodeval1.get(lnodeid)
                else:
                    lnodeobj = objvalue

                if rnodeid in nodeval:
                    rnodeobj = nodeval.get(rnodeid)
                elif rnodeid in nodeval1:
                    rnodeobj = nodeval1.get(rnodeid)
                else:
                    rnodeobj = objvalue

                if pnodeid in nodeval:
                    pnodeobj = nodeval.get(pnodeid)
                    pbranchvar = nodevar.get(pnodeid)
                elif pnodeid in nodeval1:
                    pnodeobj = nodeval1.get(pnodeid)
                    pbranchvar = nodevar1.get(pnodeid)
                else:
                    pnodeobj = objvalue
                    pbranchvar = branchvar

                lobjinc = (pnodeobj - lnodeobj)/objvalue
                robjinc = (pnodeobj - rnodeobj)/objvalue
                #存储每个分支变量对应的目标增长----------------定义方式1
                listobjinc1[branchvar].append((objvalue-bestobjvalue)/objvalue)
 
                x = relx[pbranchvar]
                #在这里整型变量的数量，即为二进制变量的数量
                thirtysix = floatchange(np.true_divide(depth-1,nint))
                thirtyseven = min(x - floor(x), ceil(x) - x)
 
               #针对变量观察到的目标增长,统计的是当前节点的数据.(用当前节点的目标值-父节点的目标值)----------------------定义方式2
                oi = (objvalue - pnodeobj)/objvalue
                #针对该变量以往观察到的目标增长，所有变量的目标增长都存在了这里面（listobjinc）
                listobjinc[branchvar].append(oi)


                #获取外面计算的第一部分特征,通过分支变量获取对应的第一部分特征
                f = f1[pbranchvar]
                #获取外面第二部分特征以及计算的第二部分特征

                #计算变量i对应的第三部分特征,通过listobjinc获取当前分支变量观察到的目标增长
                fea3 = feature3(listobjinc1[pbranchvar])
                fea31 = feature3(listobjinc[pbranchvar])
                print("fea3 = ",len(fea3),fea3)
                print("fea31 = ",len(fea31),fea31)
           
                #计算第61号特征
                totalbranchnum = self.times_called
                sixtyone = floatchange(np.true_divide(len(listobjinc1[pbranchvar]),totalbranchnum))
                #计算score，衡量分支标准的方式，两种. 
                if len(listobj1) > 1:
                    listscore = []
                    a1 = listobj1[-1]
                    a2 = listobj1[-2]
                    score1 = (a1*a2)/pnodeobj
                else:
                    score1 = listobj1[-1]/pnodeobj

                relativescore1 = score(lobjinc,robjinc)
                relativescore2 = lobjinc*robjinc
                #输出该节点的信息
                infeasum = self.get_infeasibility_sum(nodeid)
                numinfeas = self.get_num_infeasibilities(nodeid)
                print("infeasum,numinfeas",infeasum,numinfeas)
                listall.append(nodeid[0])
                listall.append(pnodeid)
                listall.append(depth)
                listall.append(branchvar)
                listall.append(objvalue)
                listall.append(pnodeobj)
                listall.append(nodenumber)
                listall.append(objvalue)
                listall.append(thirtysix)
                listall.append(thirtyseven)
                listall.append(fea3)
                listall.append(sixtyone)
                listall.append(score1)
#                print ("未explore节点数据= ",listall)
#                print("relativescore1 = ",relativescore1,score1)
#                print("relativescore2 = ",relativescore2,score1)
#                print("nodeval = ",len(nodeval),nodeval)
#                print("nodevarval = ",len(nodevarval),nodevarval)
#                print("listobjinc2 = ",branchvar,len(listobjinc2[branchvar]),listobjinc2) 

#                print(f[0],",",f[1],",",f[2],",",f[3],",",f[4],",",f[5],",",f[6],",",f[7],",",f[8],",",f[9],",",f[10],",",f[11],",",f[12],",",f[13],",",f[14],",",f[15],",",f[16],",",f[17],",",f[18],",",f[19],",",f[20],",",f[21],",",f[22],",",f[23],",",f[24],",",f[25],",",f[26],",",f[27],",",f[28],",",f[29],",",f[30],",",f[31],",",f[32],",",f[33],",",f[34],",",thirtysix,",",thirtyseven,",",listdp[1][0],",",listdp[1][1],",",listdp[1][2],",",listdp[1][3],",",listdp[1][4],",",listdp[1][5],",",listobjsa[pbranchvar][0],",",listobjsa[pbranchvar][1],",",listobjsa[pbranchvar][2],",",listobjsa[pbranchvar][3],",",listobjsa[pbranchvar][4],",",listrhssa[0][0],",",listrhssa[0][1],",",listrhssa[0][2],",",listrhssa[0][3],",",listrhssa[0][4],",",slacks[0],",",duals[0],",",fea3[0],",",floatchange(fea3[1]),",",fea3[2],",",fea3[3],",",sixtyone,",",fea3[4],",",fea3[5],",",fea3[6],",",fea3[7],",",relativescore2,file = doc)
                print(f[0],",",f[1],",",f[2],",",f[3],",",f[4],",",f[5],",",f[6],",",f[7],",",f[8],",",f[9],",",f[10],",",f[11],",",f[12],",",f[13],",",f[14],",",f[15],",",f[16],",",f[17],",",f[18],",",f[19],",",f[20],",",f[21],",",f[22],",",f[23],",",f[24],",",f[25],",",f[26],",",f[27],",",f[28],",",f[29],",",f[30],",",f[31],",",f[32],",",f[33],",",f[34],",",thirtysix,",",thirtyseven,",",floatchange(fea3[0]),",",floatchange(fea3[1]),",",floatchange(fea3[2]),",",floatchange(fea3[3]),",",sixtyone,",",floatchange(fea3[4]),",",floatchange(fea3[5]),",",floatchange(fea3[6]),",",floatchange(fea3[7]),",",relativescore2,file = doc)
#        print("listobjinc = ",len(listobjinc),listobjinc)
#        print("objinc = ",len(objinc),objinc)
#        print("nodeval = ",len(nodeval),nodeval)
       
        print("2Node-----------------------------------------------------")
        print ("3Node--------------------------------------------------------------------------------------------------------")


def admipex1(filename):
    c = CPX.Cplex(filename)
    # sys.stdout is the default output stream for log and results
    # so these lines may be omitted
    c.set_log_stream(sys.stdout)
    c.set_results_stream(sys.stdout)
    solve_instance = c.register_callback(MySolve)
    solve_instance.times_called = 0
    branch_instance = c.register_callback(MyBranch)
    branch_instance.times_called = 0
    branch_instance.cpx = c
    node_instance = c.register_callback(MyNode)
    node_instance.times_called = 0
#    c.parameters.timelimit.set(600)
    c.parameters.mip.interval.set(1)
    c.parameters.preprocessing.linear.set(0)
    c.parameters.mip.strategy.variableselect.set(3)
    c.parameters.mip.strategy.search.set(
        c.parameters.mip.strategy.search.values.traditional)
    c.parameters.threads.set(1)
    c.parameters.mip.display.values.none
    c.parameters.mip.strategy.heuristicfreq.set(-1)
    c.parameters.mip.cuts.mircut.set(-1)
    c.parameters.mip.cuts.implied.set(-1)
    c.parameters.mip.cuts.gomory.set(-1)
    c.parameters.mip.cuts.flowcovers.set(-1)
    c.parameters.mip.cuts.pathcut.set(-1)
    c.parameters.mip.cuts.liftproj.set(-1)
    c.parameters.mip.cuts.zerohalfcut.set(-1)
    c.parameters.mip.cuts.cliques.set(-1)
    c.parameters.mip.cuts.covers.set(-1)
    c.parameters.mip.cuts.disjunctive.set(-1)
    c.parameters.mip.cuts.gubcovers.set(-1)
    c.parameters.mip.cuts.localimplied.set(-1)
    c.parameters.mip.cuts.mcfcut.set(-1)
    c.parameters.mip.cuts.rlt.set(-1)
    c.parameters.mip.cuts.bqp.set(-1)
    

    c.solve()
    solution = c.solution
    # solution.get_status() returns an integer code
    print("Solution status = ", solution.get_status(), ":", end=' ')
    # the following line prints the corresponding string
    print(solution.status[solution.get_status()])
    print("Objective value = ", solution.get_objective_value())
    print()
    x = solution.get_values(0, c.variables.get_num() - 1)
    for j in range(c.variables.get_num()):
        if fabs(x[j]) > 1.0e-10:
            print("Column %d: Value = %17.10g" % (j, x[j]))
    solution.write("./sol/collect.sol")
   
    '''
    print("Solve callback was called ", solve_instance.times_called, "times")
    print("Branch callback was called ", branch_instance.times_called, "times")
    print("Node callback was called ", node_instance.times_called, "times")
    '''

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: admipex1.py filename")
        print("  filename   Name of a file, with .mps, .lp, or .sav")
        print("             extension, and a possible, additional .gz")
        print("             extension")
        sys.exit(-1)
    admipex1(sys.argv[1])

doc.close()
