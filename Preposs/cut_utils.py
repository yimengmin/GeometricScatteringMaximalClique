#adapt from https://github.com/Stalence/erdos_neu
import math
import networkx as nx
import numpy as np
import time
import gurobipy as gp
from gurobipy import GRB

def solve_gurobi_maxclique(nx_graph, time_limit = None):

    nx_complement = nx.operators.complement(nx_graph)
    x_vars = {}
    m = gp.Model("mip1")
    m.params.OutputFlag = 0

    if time_limit:
        m.params.TimeLimit = time_limit

    for node in nx_complement.nodes():
        # Create a new model

        # Create variables
        x_vars['x_'+str(node)] = m.addVar(vtype=GRB.BINARY, name="x_"+str(node))

    count_edges = 0
    for edge in nx_complement.edges():
        m.addConstr(x_vars['x_'+str(edge[0])] + x_vars['x_'+str(edge[1])] <= 1,'c_'+str(count_edges))
        count_edges+=1
    # Set objective
    m.setObjective(sum([x_vars['x_'+str(node)] for node in nx_complement.nodes()]), GRB.MAXIMIZE);


    # Optimize model
    m.optimize();

    set_size = m.objVal;
    x_vals = [var.x for var in m.getVars()] 

    return set_size, x_vals



