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
from math import floor, fabs,log,ceil
import cplex as CPX
import cplex.callbacks as CPX_CB
import sys
from random import choice


def admipex1(filename):
    c = CPX.Cplex(filename)
    # sys.stdout is the default output stream for log and results
    # so these lines may be omitted
    c.set_log_stream(sys.stdout)
    c.set_results_stream(sys.stdout)
#    c.parameters.timelimit.set(600)
    c.parameters.mip.interval.set(1)
    c.parameters.preprocessing.linear.set(0)
    #设置变量选择策略
    c.parameters.mip.strategy.variableselect.default
    #设置搜索策略
#    c.parameters.mip.strategy.search.set(c.parameters.mip.strategy.search.values.traditional)
    #设置使用的线程数量
#    c.parameters.threads.set(1)
    #设置是否打印值信息
#    c.parameters.mip.display.values.none
    #禁用相关cuts方法

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
    solution.write("./sol/cplex.sol")
#    solution.write("./sol/cplexnw.sol")
#    solution.write("./sol/cplexfast.sol")
#    print("Branch callback was called ", branch_instance.times_called, "times")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: admipex1.py filename")
        print("  filename   Name of a file, with .mps, .lp, or .sav")
        print("             extension, and a possible, additional .gz")
        print("             extension")
        sys.exit(-1)
    admipex1(sys.argv[1])
