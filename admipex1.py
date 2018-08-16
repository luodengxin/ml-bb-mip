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
from math import floor, fabs,log,ceil
from functools import reduce

import cplex as CPX
from cplex.callbacks import BranchCallback,NodeCallback,SolveCallback,IncumbentCallback
from cplex._internal._subinterfaces import SensitivityInterface,BasisInterface,AdvancedSolutionInterface,SolutionInterface
from cplex.exceptions import CplexError

import sys
import numpy as np
from userpython import feature1
#from userpython import cab
from random import choice
import time

filename = "/root/pro/project/cplexpython/lp/aflow30a.mps.gz"
c1 = CPX.Cplex()
c1.read(filename)


nint = c1.variables.get_num_integer()
nbin = c1.variables.get_num_binary()
nvar = c1.variables.get_num()
vars = c1.variables.get_names()
cons = c1.linear_constraints.get_names()


#创建一个全局变量，以字典形式存储其中产生的每个nodeid和其对应的value
nodeval = {}
#创建一个全局变量，以字典形式存储，统计每个变量观察到的目标增长
objinc = {}
#创建一个全局变量，含义是总分支次数
totalbranchnum = 0
#创建一个nvars的列表,用来存储针对每个变量观察到的目标增长
listobjinc = [[] for i in range(nvar)]

#收集灵敏度分析和DP 的静态特征
clone = CPX.Cplex(c1)
clone.set_problem_type(clone.problem_type.LP)
clone.solve()
relx = clone.solution.get_values()
basic = clone.solution.basis.get_basis()
b = [i for i,v  in enumerate(basic[0]) if v == 1 ]
dp = clone.solution.advanced.get_Driebeek_penalties(b)
objsa = clone.solution.sensitivity.objective()
rhssa = clone.solution.sensitivity.rhs()
slacks = clone.solution.get_linear_slacks()
duals = clone.solution.get_dual_values()
print(relx,len(relx))
print(dp,len(dp))
print(objsa,len(objsa))
print(rhssa,len(rhssa))
print(slacks,len(slacks))
print(duals,len(duals))

#计算第一部分特征
#用来获取c,a,b
listc = c1.objective.get_linear()

lista = []
m = c1.linear_constraints.get_num()
n = c1.variables.get_num()
for i in range(m):
    lista1 = []
    for j in range(n):
        aeff = c1.linear_constraints.get_coefficients(i,j)
        lista1.append(aeff)
    lista.append(lista1)

listb = c1.linear_constraints.get_rhs()
listv = c1.variables.get_names()
start = time.clock()
#计算第一部分特征
f1 = feature1.f1(listc,lista,listb)
end = time.clock()
print(end - start,"ms")
print("f1 = ",len(f1),f1)

print("nint1 = ",nint)
print("nbin1 = ",nbin)
print("nvar1 = ",nvar)
print("lenvar1 = ",len(vars))
print("lencons1 = ",len(cons))



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

  
def get_candiate_branching_variable(solution):
    b = [i for i,v  in enumerate(solution) if not v.is_integer() ]
    return b


#获得dp
def Driebeek_penalty(basic_var):
    return dp;


#获得灵敏度分析
def Sensitivity_A(candiate_var):
    return sa;






class MySolve(SolveCallback):
    def __call__(self):
        self.times_called += 1
        if self.get_num_nodes() < 1:
            self.solve(self.method.primal)
        else:
            self.solve(self.method.dual)
        status = self.get_cplex_status()
        self.use_solution()


class MyIncumbent(IncumbentCallback):
    def __call__(self):
        a = self.get_solution_source()
        print("a = ",a)

def ff():
    return "hello world";

#为了收集候选分支变量
results = []
class MyBranch1(BranchCallback, NodeCallback, SensitivityInterface, BasisInterface, AdvancedSolutionInterface, SolutionInterface, CplexError):
    
    c1 = CPX.Cplex()
    def __init__(self, env):
        BranchCallback.__init__(self, env)
#        super(NodeCallback, self).__init__(env)
#        NodeCallback.__init__(self, env)
        

    def __call__(self):
        c = CPX.Cplex()
        self.times_called += 1
        nrows = self.get_num_rows()
        ncols = self.get_num_cols()
        print("nrows,ncols = ",nrows,ncols)
        br_type = self.get_branch_type()
        self.get_objective_value()
        print ("brtype = ",br_type)
        if (br_type == self.branch_type.SOS1 or
                br_type == self.branch_type.SOS2):
            return
        #获取当前节点的变量的值
        x = self.get_values()
        objval = self.get_objective_value()
        obj = self.get_objective_coefficients()
        feas = self.get_feasibilities()
        #判断当前节点是否是整数可行的
        nodestatus = self.is_integer_feasible()
        if nodestatus == True:
            s= "MIP"
        else:
            s= "LP"

        #变量不可行状态(1)
        status1 = self.feasibility_status.infeasible
        #变量可行状态(0)
        status2 = self.feasibility_status.feasible

        print("status1,status2 = ",status1,status2)
        #用来获取当前解中所有的分数变量包含所有的不可行变量

        #用来获取所有的不可行变量
        candiatevar = []
        for i in range(len(x)):
            if feas[i] == status1:
                candiatevar.append(i)
      
#        print("candiatevar = ",len(candiatevar),candiatevar)
        if len(candiatevar) == 0:
            return

        #为了测试外面的方法使用
        '''
        clone = CPX.Cplex(c1)
        clone.set_problem_type(clone.problem_type.LP)
        clone.solve()
        relx = clone.solution.get_values()
        basic = clone.solution.basis.get_basis()
        b = [i for i,v  in enumerate(basic[0]) if v == 1 ]
        dp = clone.solution.advanced.get_Driebeek_penalties(b)
        objsa = clone.solution.sensitivity.objective()
        rhssa = clone.solution.sensitivity.rhs()
        slacks = clone.solution.get_linear_slacks()
        duals = clone.solution.get_dual_values()
        #self.SOS.type.SOS1不可行
        sostype = self.branch_type.SOS2
        solvemethod = self.method.MIP
#        print("candiatevar = ",len(candiatevar),len(b),candiatevar,b,sostype,solvemethod)
        print(relx,len(relx))
        print(dp,len(dp))
        print(objsa,len(objsa))
        print(rhssa,len(rhssa))
        print(slacks,len(slacks))
        print(duals,len(duals))
        clone.solution.write("./sol/admip1.sol")
        '''


        hello = ff() 
#        print("hello = ",hello)
        print ("0Branch--------------------------------------------------------------------------------------------------------")
        branchnums = self.get_num_branches()
        i=0 
        while i >=0 and i < branchnums:
            branchvar = self.get_branch(i)
            var1 = branchvar[1][0][0]
            print("当前节点的候选分支(estimatedobjval,(var,dir,bnd)) = ",branchvar,var1)
            i = i+1
        #为了将候选分支变量传递给NodeCallback使用
        results.append(var1)
        nodeid = self.get_node_ID()
        nodedata = self.get_node_data()
        hasincumbent = self.has_incumbent()
        bestobjval = self.get_best_objective_value()
        incumbentobjval = self.get_incumbent_objective_value()
        cutoff = self.get_cutoff()
        list1 = []
        list1.append(s)
        list1.append(nodeid)
        list1.append(nodedata)
        list1.append(hasincumbent)
        list1.append(cutoff)
        print(len(list1),list1)
        print("objval,bestobjval,incumbentobjval = ",objval,bestobjval,incumbentobjval)
  
        nodes = self.get_num_nodes()
        remaining_nodes = self.get_num_remaining_nodes()
        print("Nnodes,Rnodes = ",nodes,remaining_nodes)

class MyNode(NodeCallback):
    def __call__(self):
        self.times_called += 1
        print("1Node-------------------------------------------------------")
        nodes = self.get_num_nodes()
        remaining_nodes = self.get_num_remaining_nodes()
        print("Nnodes,Rnodes = ",nodes,remaining_nodes)

        bv = results[-1]
        print("results = ",bv)
        bestobjvalue = self.get_best_objective_value()
        incumbentobjvalue = self.get_incumbent_objective_value()
        varvalues = self.get_incumbent_values()
        nodeval[0]=bestobjvalue 
        print("bestobj,incumbentobj= ",bestobjvalue,incumbentobjvalue)
        sbscore = 1.0
        pnodeid = 0
        nodeid = 0
        for node in range(self.get_num_remaining_nodes()):
            listall = []
            #当前节点id，因为是唯一标识的，所以在整个BB树中，只有唯一的一个
            nodeid = self.get_node_ID(node)
            #当前节点的父节点的ID，因为是唯一标识的，所以在整个BB树中，只有唯一的一个(因为节点ID是从0开始标识的，所以必须(向上取整-1)才可。)
            pnodeid = np.ceil(nodeid[0]/2) - 1
  
            nodenumber = self.get_node_number(nodeid)
            #求得当前节点的分支变量
            branchvar = self.get_branch_variable(nodeid)
            #求得当前节点的深度
            depth = self.get_depth(nodeid)
            #求得父节点的深度
            pdepth = depth - 1
            nodedata = self.get_node_data(nodeid)
            #求得当前节点的目标值
            objvalue = self.get_objective_value(nodeid)
            #往字典nodeval里面存数据,字典原理：键必须唯一，值可以不唯一（每个几点只要它的nodeid相同，那么objvalue一定相同）。将BB树中产生的所有节点ID以及对应的目标值存进字典中            #key：nodeid;value:objvalue
            nodeval[nodeid[0]] = objvalue
            #计算当前节点的上一层的数目
            #计算当前节点的上一层之前的节点数目（即当前节点的上上层，包含上上层之前的节点数目之和）
            #求得父节点的目标值(从nodeval-存储目前产生的所有节点信息，依据父节点ID，得到对应的目标值)
            if pnodeid in nodeval:
                pobjvalue = nodeval.get(pnodeid)
            else:
                pobjvalue = objvalue

            x = varvalues[branchvar]
            #在这里整型变量的数量，即为二进制变量的数量
            thirtysix = depth/nbin
            thirtyseven = min(x - floor(x), ceil(x) - x)
            print("pnodeid = ",pnodeid)
            #往字典objinc里面存，针对变量观察到的目标增长,统计的是当前节点的数据.(用当前节点的目标值-父节点的目标值)
            oi = objvalue - pobjvalue
            
            #针对该变量以往观察到的目标增长，所有变量的目标增长都存在了这里面（listobjinc）            
            listobjinc[branchvar].append(oi)
            #然后以字典的形式全部存在了objinc,key：变量i，value：listobjinc，一个列表，里面存储的是针对当前变量i观察到的目标增长
            objinc[branchvar] = listobjinc
            #获取外面计算的第一部分特征
            f = f1[branchvar]
            #获取外面第二部分特征以及计算的第二部分特征
            #计算变量i对应的第三部分特征
            fea3 = feature3(listobjinc[branchvar])
            totalbranchnum = self.times_called
            #计算第61号特征
            sixtyone = len(listobjinc[branchvar])/totalbranchnum
            
            if len(listobjinc[branchvar]) > 1:
                score1 = listobjinc[branchvar][-1] * listobjinc[branchvar][-2]
            else:
                score1 = listobjinc[branchvar][-1]
                
            score2 = reduce(lambda x,y:x*y,listobjinc[branchvar])
            if branchvar == bv:
                sbscore *= ((incumbentobjvalue - objvalue)/objvalue) 

            #往全局变量nodeval里面添加元素 
            listall.append(branchvar)
            listall.append(nodeid[0])
            listall.append(objvalue)
            listall.append(pnodeid)
            listall.append(pobjvalue)
            listall.append(type(pobjvalue)) 
            listall.append(node)
            listall.append(nodenumber)
            listall.append(depth)
            listall.append(nodedata)
            listall.append(objvalue)
            listall.append(thirtysix)
            listall.append(thirtyseven)
            listall.append(fea3)
            listall.append(sixtyone)        
            listall.append(score1)
            listall.append(score2)
            print ("未explore节点数据= ",listall)
        print("nodeval = ",len(nodeval),nodeval)
        print("2Node-----------------------------------------------------")
        print ("3Node--------------------------------------------------------------------------------------------------------")

def admipex1(filename):
    try:
        c = CPX.Cplex(filename)
        #solve relaxation

        # sys.stdout is the default output stream for log and results
        # so these lines may be omitted
        c.set_log_stream(sys.stdout)
        c.set_results_stream(sys.stdout)
        solve_instance = c.register_callback(MySolve)
        solve_instance.times_called = 0
        branch_instance1 = c.register_callback(MyBranch1)
        branch_instance1.cpx = c
        branch_instance1.times_called = 0
        node_instance = c.register_callback(MyNode)
        node_instance.times_called = 0
        incumbent_instance = c.register_callback(MyIncumbent)
        c.parameters.mip.interval.set(1)
        c.parameters.preprocessing.linear.set(0)
        c.parameters.mip.strategy.variableselect.set(3)
        c.parameters.mip.strategy.search.set(
        c.parameters.mip.strategy.search.values.traditional)
        c.parameters.mip.display.values.none
        #solve relaxation
        '''
        clone = CPX.Cplex(c)
        clone.set_problem_type(clone.problem_type.LP)
        clone.solve()
        relx = clone.solution.get_values()
        basic = clone.solution.basis.get_basis()
        b = [i for i,v  in enumerate(basic[0]) if v == 1 ]
        dp = clone.solution.advanced.get_Driebeek_penalties(b)
        objsa = clone.solution.sensitivity.objective()
        rhssa = clone.solution.sensitivity.rhs()
        slacks = clone.solution.get_linear_slacks()
        duals = clone.solution.get_dual_values()
        print(relx,len(relx))
        print(dp,len(dp))
        print(objsa,len(objsa))
        print(rhssa,len(rhssa))
        print(slacks,len(slacks))
        print(duals,len(duals))
        clone.solution.write("./sol/admip1.sol")
        '''


        '''
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
        '''
        c.solve()
    except CplexError as exc:
        print(exc)
        return

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


    print("len(nodeval) = ",len(nodeval))
    print("len(objinc) = ",len(objinc))
    print("len(listobjinc) = ",len(listobjinc))
    solution.write("./sol/admipex1.sol")
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
