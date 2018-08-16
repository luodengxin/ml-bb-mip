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

#ML相关包
from sklearn import metrics
from sklearn import preprocessing
from sklearn.preprocessing import  Imputer
from sklearn import metrics
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.externals import joblib
#load model
print("MODEL is stand style")
model = joblib.load('/root/pro/model/bp12')


#用来存储收集到的数据集
doc = open("/root/pro/data/BP102_10.txt",'a')
filename = "/root/pro/project/cplexpython/lp/aflow30a.mps.gz"
c1 = CPX.Cplex()
c1.read(filename)
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
#创建一个全局变量，用来存储对应节点的深度
nodedepth = {}
#创建一个全局变量，以字典形式存储其中产生的每个nodeid和其对应的objinc
nodeoi = {}
#创建一个全局变量，以字典形式存储，统计每个变量观察到的目标增长
objinc = {}
#创建一个全局变量，含义是总分支次数
totalbranchnum = 0
#创建一个nvars的列表,用来存储针对每个变量观察到的目标增长
listobjinc = [[] for i in range(nvar)]
listobjinc1 = [[] for i in range(nvar)]
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
n = lc.get_num()
ma = np.zeros([m,n])
for i in range(m):
    for j in range(n):
        ma[i][j] = lc.get_coefficients(i,j)

lista = list(ma) 
  
 
listb = lc.get_rhs()
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
    if len(l) > 0:
        l4 = np.array(l,dtype = 'float_')
        mean = np.mean(l4)
        std  = np.std(l4,ddof = 1)
        min  = np.min(l4)
        max  = np.max(l4)
        l4.sort()
        q = np.percentile(l4,[25, 50, 75, 100])
        a1 = q[0]
        a2 = q[1]
        a3 = q[2]
        a4 = q[3]
    else:
        mean = 0 
        std = 0 
        min = 0 
        max = 0 
        a1 = 0 
        a2 = 0 
        a3 = 0 
        a4 = 0 
    fea.append(mean)
    fea.append(std)
    fea.append(min)
    fea.append(max)
    fea.append(a1)
    fea.append(a2)
    fea.append(a3)
    fea.append(a4)
    return fea;
#a代表左节点的目标增长，b代表右节点的目标增长
def score(a,b):
    e = 10**-6
    return max(a,e)*max(b,e)



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


listobj = []

class MyBranch(BranchCallback, NodeCallback1,NodeCallback, SensitivityInterface,BasisInterface,AdvancedSolutionInterface,SolutionInterface):

    def one(l):
        return min(l)
  

    def __call__(self):
        self.times_called += 1
        #获取当前节点中的所有数据 
        bestobjval = self.get_best_objective_value()
        incumbentobjval = self.get_incumbent_objective_value()
        objval = self.get_objective_value()
        nodeid = self.get_node_ID()
        nodedata = self.get_node_data()
        hasincumbent = self.has_incumbent()
        nodes = self.get_num_nodes()
        remaining_nodes = self.get_num_remaining_nodes()
        objval = self.get_objective_value()
        x = self.get_values()
        #存储当前所有变量的状态，0表示可行，1表示不可行 
        feas = self.get_feasibilities()
        #判断当前节点是否是整数可行的
        boolean = self.is_integer_feasible()
        #可以加入的其他特征
        pseudocosts = self.get_pseudo_costs()
        linearslacks = self.get_linear_slacks()
        lowerbounds = self.get_lower_bounds()
        upperbounds = self.get_upper_bounds()
        #用来存储每次分支前后的最优解
        listobj.append(bestobjval)

        print("测试方法的可用性 = ",self.objsa())
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
#        candiatevar = [i for i,v  in enumerate(feas) if v == 1 ]

        nodedepth[0] = 1
        thirtysix = nodedepth[nodeid]/nbin
        totalbranchnum = self.times_called
         
        #用来获取索引r的
        if 0 in linearslacks:
            r = linearslacks.index(0)
        else:
            r = 0

        print("0branch----------------------------------------------------------------------------------------------------------")
        #单独用来存储BC中产生的节点
        print("b,i,o = ",bestobjval,incumbentobjval,objval)

        #变量不可行状态(1)-----对应于basic
        status1 = self.feasibility_status.infeasible
        #变量可行状态(0)------对应于at_lower_bound
        status2 = self.feasibility_status.feasible
        #用来存储所有候选分支变量的对应索引
        candiatevar = []
        #用来存储所有候选分支变量的对应score
        ss = []
   
        maxobj = -CPX.infinity
        maxinf = -CPX.infinity
        bestj = -1
        obj = self.get_objective_coefficients()

        #获得候选分支变量,对这些候选分支变量计算相应的特征，从中选择一个好的变量作为分支变量，进行分支
        for i in range(len(x)):
            if feas[i] == status1:
                #可以对candiatevar进行Driebeek penalties
                #可以对candiatevar进行目标函数系数灵敏度分析
                #可以对candiatevar对应约束行进行右边侧系数灵敏度分析

                #用来存储所有候选分支变量对应的索引
                candiatevar.append(i)
                varval = x[i]
                thirtyfive = f1[i]
                #将前35个特征一个一个存进feature集中
                feature = [thirtyfive[j] for j,v in enumerate(thirtyfive)]
               
                thirtyseven =  min(varval - floor(varval),ceil(varval) - varval)
                sixtyone = len(listobjinc[i])/totalbranchnum
                fea3 = feature3(listobjinc1[i])
                '''
                thirtyfive.append(thirtysix)
                thirtyfive.append(thirtyseven)
                thirtyfive.append(fea3[0])
                thirtyfive.append(floatchange(fea3[1]))
                thirtyfive.append(fea3[2])
                thirtyfive.append(fea3[3])
                thirtyfive.append(sixtyone)
                thirtyfive.append(fea3[4])
                thirtyfive.append(fea3[5])
                thirtyfive.append(fea3[6])
                thirtyfive.append(fea3[7])
                '''
                feature.append(thirtysix)
                feature.append(thirtyseven)
#                feature.append(floatchange(listdp[1][0]))
#                feature.append(floatchange(listdp[1][1]))
#                feature.append(floatchange(listdp[1][2]))
#                feature.append(floatchange(listdp[1][3]))
#                feature.append(floatchange(listdp[1][4]))
#                feature.append(floatchange(listdp[1][5]))
#                feature.append(floatchange(listobjsa[0][0]))
#                feature.append(floatchange(listobjsa[0][1]))
#                feature.append(floatchange(listobjsa[0][2]))
#                feature.append(floatchange(listobjsa[0][3]))
#                feature.append(floatchange(listobjsa[0][4]))
#                feature.append(floatchange(listrhssa[0][0]))
#                feature.append(floatchange(listrhssa[0][1])) 
#                feature.append(floatchange(listrhssa[0][2])) 
#                feature.append(floatchange(listrhssa[0][3])) 
#                feature.append(floatchange(listrhssa[0][4]))
#                feature.append(floatchange(slacks[0]))
#                feature.append(floatchange(duals[0]))
                feature.append(fea3[0])
                feature.append(floatchange(fea3[1]))
                feature.append(fea3[2])
                feature.append(fea3[3])
                feature.append(sixtyone)
                feature.append(fea3[4])
                feature.append(fea3[5])
                feature.append(fea3[6])
                feature.append(fea3[7])
#                print("feature大小 = ",len(feature),feature)
                feature = np.array(feature).reshape((1,-1))
                """
                #使用随机生成的测试数据集
                testx = np.random.rand(63)
                testx = list(testx)
                testx = np.array(testx).reshape((1,-1))
                """
             
                #使用训练好的模型根据每个候选分支变量计算的特征预测相对应的sbscore
                score = model.predict(feature)
                #用来存储所有候选分支变量对应的score
                ss.append(score)

#        if bestj < 0:
#            return
 
        #得到score最大的那个变量对应的索引，进行分支
        if len(ss) > 0:
            index = ss.index(max(ss))
            bestj = candiatevar[index]
        else:
#            bestj = bestj
             return

        xj_lo = floor(x[bestj])
        # the (bestj, xj_lo, direction) triple can be any python object to
        # associate with a node
        self.make_branch(objval, variables=[(bestj, "L", xj_lo + 1)],
                          node_data=(bestj, xj_lo, "UP"))
        self.make_branch(objval, variables=[(bestj, "U", xj_lo)],
                          node_data=(bestj, xj_lo, "DOWN"))

listobj1 = []
class MyNode(NodeCallback,NodeCallback1):
    def __call__(self):
        self.times_called += 1
        print("1Node--------------------------------------------------------------------------------------------------------------")
        nodes = self.get_num_nodes()
        remaining_nodes = self.get_num_remaining_nodes()
        bestobjvalue = self.get_best_objective_value()
        incumbentobjvalue = self.get_incumbent_objective_value()
        print("Nnodes,Rnodes = ",nodes,remaining_nodes)
        print("bestobj,incumbentobj= ",bestobjvalue,incumbentobjvalue,self.has_incumbent())
        for node in range(self.get_num_remaining_nodes()):
            nodeid = self.get_node_ID(node)
            if nodeid[0] not in nodedepth:
                branchvar = self.get_branch_variable(nodeid)
                depth = self.get_depth(nodeid)
                objvalue = self.get_objective_value(nodeid)
                nodeval[nodeid[0]] = objvalue
                nodevar[nodeid[0]] = branchvar
                nodedepth[nodeid[0]] = depth
                listobjinc1[branchvar].append((objvalue-bestobjvalue)/objvalue)
        print ("2Node--------------------------------------------------------------------------------------------------------")


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
    c.parameters.timelimit.set(600)
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
    solution.write("./sol/mlibcut.sol")
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
