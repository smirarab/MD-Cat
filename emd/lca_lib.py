from treeswift import *
import logging
from sys import stdout

logger = logging.getLogger("lca_lib")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(stdout)
formatter = logging.Formatter('%(levelname)s:%(name)s:%(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.propagate = False

def find_LCAs(myTree,myQueries):
# find LCA for each of the list of nodes in myQueries
# return the label of the LCA node of each query
# based on the algorithm described here https://cp-algorithms.com/graph/lca.html
    def euler_tour():
    # perform the euler_tour (DFS) on the tree
    # output: 
    #   + E: the euler tour
    #   + F: the index in E where each node first occurs
    #   + H: the topological depth of each node (root depth is 1)
        root = myTree.root
        E = [root]
        H = {root: 1}
        F = {root.label: 0}
        duplicates = set()
        stack = [(root, iter(root.children))]
        while stack:
            node, children = stack[-1]
            child = next(children, None)
            if child is None:
                stack.pop()
                if stack:
                    # Revisit the parent after completing each child subtree.
                    E.append(stack[-1][0])
            else:
                if child.label in F:
                    duplicates.add(child.label)
                F[child.label] = len(E)
                H[child] = H[node] + 1
                E.append(child)
                stack.append((child, iter(child.children)))
        return E,F,H,duplicates
    def min_segment_tree(E,H):
    # build a min segment-tree
    # to query the minimum of any range of H
    # in O(logn)
        n = len(E)
        t = [None]*(4*n)
        def __build__(node,b,e):
            if b==e:
                t[node] = E[b]
            else:
                mid = (b+e)//2
                __build__(2*node,b,mid) 
                __build__(2*node+1,mid+1,e)
                l = t[2*node]
                r = t[2*node+1]
                t[node] = l if H[l] < H[r] else r
        __build__(1,0,n-1)     
        return t                          

    def query_segment_tree(t,q,E,F,H):
        def __query__(node,b,e,L,R):
            if (R < b or L > e):
                return None
            if (b >= L and e <= R):
                return t[node]    
            mid = (b+e)//2
            left = __query__(node*2,b,mid,L,R)
            right = __query__(node*2+1,mid+1,e,L,R)
            if left is None:
                return right
            if right is None:
                return left
            return left if H[left] < H[right] else right        

        if not q:
            raise ValueError("calibration query must contain at least one node name")
        missing = [a for a in q if a not in F]
        if missing:
            raise ValueError(
                "calibration query " + "+".join(q)
                + ": node name(s) not found in the input tree: "
                + ", ".join(missing))

        ambiguous = sorted(set(q) & duplicates)
        if ambiguous:
            raise ValueError(
                "calibration query " + "+".join(q)
                + ": ambiguous duplicate node name(s) in the input tree: "
                + ", ".join(ambiguous))

        L = None
        R = None
        for a in q:
            L = min(F[a],L) if L is not None else F[a]
            R = max(F[a],R) if R is not None else F[a]
        return __query__(1,0,len(E)-1,L,R)

    E,F,H,duplicates = euler_tour()
    t = min_segment_tree(E,H)
    myLCAs = []
    for q in myQueries:
        lca = query_segment_tree(t,q,E,F,H)
        myLCAs.append(lca)
    return myLCAs    
