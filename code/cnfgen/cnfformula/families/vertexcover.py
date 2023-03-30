#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""Formulas that encode coloring related problems
"""


from cnfformula.cnf import CNF
from cnfformula.cmdline import SimpleGraphHelper

from cnfformula.cmdline  import register_cnfgen_subcommand
from cnfformula.families import register_cnf_generator

from cnfformula.graphs import enumerate_vertices,enumerate_edges,neighbors
from itertools import combinations,combinations_with_replacement,product


@register_cnf_generator
def VertexCover(G,k, alternative = False):
    r"""Generates the clauses for a vertex cover for G of size <= k

    Parameters
    ----------
    G : networkx.Graph
        a simple undirected graph
    k : a positive int
        the size limit for the vertex cover

    Returns
    -------
    CNF
       the CNF encoding for vertex cover of size :math:`\leq k` for graph :math:`G`

    """
    F=CNF()

    if not isinstance(k,int) or k<1:
        ValueError("Parameter \"k\" is expected to be a positive integer")

    # Describe the formula
    name="{}-vertex cover".format(k)

    if hasattr(G,'name'):
        F.header=name+" of graph:\n"+G.name+".\n\n"+F.header
    else:
        F.header=name+".\n\n"+F.header

    # Fix the vertex order
    V=enumerate_vertices(G)
    E=enumerate_edges(G)

    def D(v):
        return "x_{{{0}}}".format(v)

    def M(v,i):
        return "g_{{{0},{1}}}".format(v,i)

    def N(v):
        return tuple(sorted([ v ] + [ u for u in G.neighbors(v) ]))

    # Create variables
    for v in V:
        F.add_variable(D(v))
    for i,v in product(range(1,k+1),V):
        F.add_variable(M(v,i))

    # No two (active) vertices map to the same index
    for i in range(1,k+1):
        for c in CNF.less_or_equal_constraint([M(v,i) for v in V],1):
            F.add_clause(c)

    # (Active) Vertices in the sequence are not repeated
    for i,j in combinations_with_replacement(range(1,k+1),2):
        i,j = min(i,j),max(i,j)
        for u,v in combinations(V,2):
            u,v = max(u,v),min(u,v)
            F.add_clause([(False,M(u,i)),(False,M(v,j))])

    # D(v) = M(v,1) or M(v,2) or ... or M(v,k)
    for i,v in product(range(1,k+1),V):
        F.add_clause([(False,M(v,i)),(True,D(v))])
    for v in V:
        F.add_clause([(False,D(v))] + [(True,M(v,i)) for i in range(1,k+1)])

    # Every neighborhood must have a true D variable
    for v1,v2 in E:
        F.add_clause([(True,D(v1)), (True,D(v2))])

    return F


@register_cnfgen_subcommand
class VertexCoverCmdHelper(object):
    """Command line helper for k-vertex cover
    """
    name='kcover'
    description='k-Vertex cover'

    @staticmethod
    def setup_command_line(parser):
        """Setup the command line options for vertex cover formula

        Arguments:
        - `parser`: parser to load with options.
        """
        parser.add_argument('k',metavar='<k>',type=int,action='store',help="size of the vertex cover")
        SimpleGraphHelper.setup_command_line(parser)


    @staticmethod
    def build_cnf(args):
        """Build the k-vertex cover formula

        Arguments:
        - `args`: command line options
        """
        G = SimpleGraphHelper.obtain_graph(args)
        return VertexCover(G, args.k)
