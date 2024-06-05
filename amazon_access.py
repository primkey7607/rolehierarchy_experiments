import dask.dataframe as dd
import pandas as pd
import os
from ast import literal_eval
import pickle
import copy

'''
Purpose: analyze the Amazon access dataset's role hierarchy,
and construct a hierarchy table of a given size based on this.
One important decision we are going to make is that, rather than using
privileges to decide the hierarchy, we will use the persons and their managers.
Since we will anyway have to generate the views and privileges ourselves,
we can enforce our assumption that persons' privileges are subsumed by managers'
privileges. That is, parents should always have fewer privileges than children
for the policy to be well-formed.

Specifically, we use the Kaggle variant of the Amazon data:
https://www.kaggle.com/competitions/amazon-employee-access-challenge/data?select=test.csv
There was also an option to use the original Amazon access dataset, but they anonymized the manager data
to the point where we can't mine any hierarchies from it (because manager IDs are entirely disjoint from PersonIDs):
https://archive.ics.uci.edu/dataset/216/amazon+access+samples
'''

class kTree:
    def __init__(self, person):
        self.children = []
        self.person = person
    
    def insert(self, new_person):
        new_tree = kTree(new_person)
        self.children.append(new_tree)
    
    def tree_append(self, other_tree):
        new_other = copy.deepcopy(other_tree)
        self.children.append(new_other)
    
    def tree2dict(self):
        if self.children == []:
            return {self.person : []}
        
        child_trees = [c.tree2dict() for c in self.children]
        return {self.person : child_trees}
    
    def max_depth(self):
        if self.children == []:
            return 1 #only the root of this tree
        
        return 1 + max([c.max_depth() for c in self.children])
    
    def min_depth(self):
        if self.children == []:
            return 1 #only the root of this tree
        
        return 1 + min([c.max_depth() for c in self.children])
    
    def num_nodes(self):
        if self.children == []:
            return 1
        return 1 + sum([c.num_nodes() for c in self.children])
    
    def contains(self, person_value):
        if self.person == person_value:
            return True
        else:
            for c in self.children:
                if c.contains(person_value):
                    return True
        
        return False

def read_amazon(fpath):
    df = dd.read_csv(fpath, sample=10000000)
    return df

#we only want the person columns of the amazon accesses, nothing else
def read_amazon_roles(fpath):
    fields = ['id', 'MGR_ID', 'ROLE_TITLE']
    df = pd.read_csv(fpath, usecols=fields)
    return df

#very important that we do this only on the smaller roles-only dataframe!
#otherwise we will break the machine...
def row_byperson(person_id, roledf):
    df_rows = roledf[roledf['id'] == person_id]
    return df_rows
    

def add_nodes(row, raw_adj_lst, roledf):
    p_id = row['id']
    mgr_id = row['MGR_ID']
    
    mgr_df = row_byperson(mgr_id, roledf)
    if mgr_df.empty:
        #doing what the print says below is actually a mistake.
        #there are people at the top of the hierarchy (otherwise it would not be a hierarchy)
        #and i think this is the root cause of the problems in my recursion below.
        #namely, we just keep looking through the whole entire graph for parents, exhausting all 
        #nodes, so there are none to choose from later.
        # print("MGR_ID is not a person, so skipping any addition here: {}".format(row))
        
        #instead, we should add the parent entry for both, knowing that
        #some managers will have no child nodes, and that makes sense.
        #and in fact, once we know this, we also know exactly where our root nodes are.
        #we don't have to start from random points in the middle.
        
        if p_id not in raw_adj_lst:
            raw_adj_lst[p_id] = {}
            raw_adj_lst[p_id]['parent'] = []
            raw_adj_lst[p_id]['child'] = []
        
        raw_adj_lst[p_id]['child'].append(mgr_id)
        
        #now, backward pass to add parents
        if mgr_id not in raw_adj_lst:
            raw_adj_lst[mgr_id] = {}
            raw_adj_lst[mgr_id]['parent'] = []
            raw_adj_lst[mgr_id]['child'] = []
        
        raw_adj_lst[mgr_id]['parent'].append(p_id)
        
        return raw_adj_lst, False
    
    #if manager exists, then update the adjacency list.
    if p_id not in raw_adj_lst:
        raw_adj_lst[p_id] = {}
        raw_adj_lst[p_id]['parent'] = []
        raw_adj_lst[p_id]['child'] = []
    
    raw_adj_lst[p_id]['child'].append(mgr_id)
    
    #now, backward pass to add parents
    if mgr_id not in raw_adj_lst:
        raw_adj_lst[mgr_id] = {}
        raw_adj_lst[mgr_id]['parent'] = []
        raw_adj_lst[mgr_id]['child'] = []
    
    raw_adj_lst[mgr_id]['parent'].append(p_id)
    
    return raw_adj_lst, True
        

def get_person_con(fpath):
    #keys are integer indices
    #each key will have a dictionary with key 'parent'
    #and a dictionary with key 'child'
    #the values of each of these dictionaries will be lists of indices adjacent to these keys.
    #we do this for both the person and managers, and transfer this mapping to business titles and details.
    #this could make things tricky, because self-loops could exist...we will see what happens.
    
    #procedure: first, construct the raw adjacency list for people
    #then, construct the role hierarchy from that using people's role codes.
    #that's cleaner than trying to do both at the same time. Also, the dataset is small enough that efficiency
    #will not be a problem.
    
    raw_adj_lst = {}
    
    df = read_amazon_roles(fpath)
    add_cnt = 0
    miss_cnt = 0
    for row in df.to_dict(orient='records'):
        raw_adj_lst, add_flag = add_nodes(row, raw_adj_lst, df)
        if add_flag:
            add_cnt += 1
        else:
            miss_cnt += 1
    
    print("Added " + str(add_cnt) + " Nodes")
    print("Missed " + str(miss_cnt) + " Nodes")
    
    with open('amazon_raw_userhierarchy.json', 'w+') as fh:
        print(raw_adj_lst, file=fh)

def more_than_two(hier : dict):
    path_cnt = 0
    all_nodes = copy.deepcopy(list(hier.keys()))
    for k in hier:
        if k not in all_nodes: #already counted
            continue
        parents = [p for p in hier[k]['parent'] if p != k]
        children = [p for p in hier[k]['child'] if p != k]
        
        ancestor_exists = False
        descendant_exists = False
        for p in parents:
            if hier[p]['parent'] != [] and hier[p]['parent'] != [p]:
                ancestor_exists = True
        
        for c in children:
            if hier[c]['child'] != [] and hier[c]['child'] != [c]:
                descendant_exists = True
        
        if ancestor_exists or descendant_exists:
            path_cnt += 1
        
        all_nodes = [a for a in all_nodes if a != k]
    
    print("Number of nodes in a path of length greater than 2: {}".format(path_cnt))
    return path_cnt
        

#Purpose: gauge whether this hierarchy is interesting.
#a hierarchy is interesting if it has a high number of paths greater than length 2.
#so, print all length-2 paths, and their count.
def analyze_hierarchy(hier_path):
    
    hier = literal_eval(open(hier_path, 'r').read())
    
    #actually, do something simple. just check for paths with length greater than 2.
    longpath_cnt = more_than_two(hier)
    print(longpath_cnt)

#whatever goes up...
def traverse_parents(tree, hier, k, all_nodes):
    hier_children = hier[k]['child']
    if hier_children == [] or hier_children == [k]:
        all_nodes = [a for a in all_nodes if a != k]
        return tree, all_nodes
    
    if len(hier_children) > 1:
        raise Exception("Data error--person should only have one manager, according to data: {}, {}".format(k, hier[k]['child']))
    
    all_nodes = [a for a in all_nodes if a != k]
    new_tree = copy.deepcopy(tree)
    
    for c in hier_children: #would be incorrect with more than one child, but we know that there is only one child.
        if c not in all_nodes:
            continue
        c_tree = kTree(c)
        c_tree.tree_append(tree)
        new_tree, all_nodes = traverse_parents(c_tree, hier, c, all_nodes)
        all_nodes = [a for a in all_nodes if a != c and a != k]
    
    return new_tree, all_nodes

#...must come down
def traverse_children(tree, hier, k, all_nodes):
    hier_parents = hier[k]['parent']
    if hier_parents == [] or hier_parents == [k]:
        all_nodes = [a for a in all_nodes if a != k]
        return tree, all_nodes
    
    all_nodes = [a for a in all_nodes if a != k]
    new_tree = copy.deepcopy(tree)
    
    for p in hier_parents:
        if p not in all_nodes:
            continue
        p_tree = kTree(p)
        inter_tree, all_nodes = traverse_children(p_tree, hier, p, all_nodes)
        new_tree.tree_append(inter_tree)
        # new_tree = copy.deepcopy(tree)
        all_nodes = [a for a in all_nodes if a != p and a != k]
    
    return new_tree, all_nodes

def extract_hierarchy(hier_path):
    '''
    Design: we can use memory, so let us just construct the trees.
    we will anyway want to do this to understand the hierarchy.
    If we construct them in a depth-first fashion, we should avoid the need to 
    merge disjoint trees we found, etc.
    
    So the procedure is: given a copy of the nodes, start with some node. follow the parents of that node.
    then follow the children of that node. remove the nodes as you follow them.
    once there are no more to follow, look at your list of remaining nodes. follow the next one.
    if we do this, we can construct a spanning forest for the graph. each tree will be a hierarchy we can use.
    
    UPDATE: the problem with this is if you traverse the whole graph when searching for parents.
    Instead, let us make use of the fact that there are managers with no manager (naturally).
    we will use these as root nodes, and find all children.
    '''
    
    hier = literal_eval(open(hier_path, 'r').read())
    roots = [k for k in hier if hier[k]['child'] == []]
    print("Roots: {}".format(roots))
    all_nodes = copy.deepcopy(list(hier.keys()))
    tree_dct = {}
    tree_cnt = 0
    for k in roots:
        if k not in all_nodes:
            continue
        init_tree = kTree(k)
        
        #NOTE: I got my semantics backward. the way this tree will have to work,
        #the root will be the person with all privileges, and the leaves will be others.
        #this is because every employee has only one manager, but a manager can have multiple employees.
        #so to find the root, we actually have to traverse the "child" nodes, which are the ACTUAL parents.
        #and the parent nodes are the ACTUAL children.
        
        #first, traverse parents--step not needed because we are already starting with roots
        # parent_tree, all_nodes = traverse_parents(init_tree, hier, k, all_nodes)
        
        #then children
        child_tree, all_nodes = traverse_children(init_tree, hier, k, all_nodes)
        
        #having exhaustively searched, add to tree dictionary
        tree_dct[tree_cnt] = child_tree
        tree_cnt += 1
    
    return tree_dct

def person2roles(fpath, tree):
    df = read_amazon_roles(fpath)

def print_tree_stats(forest):
    out_schema = ['Key', 'Max Depth', 'Min Depth', 'Nodes']
    stat_dct = {}
    for o in out_schema:
        stat_dct[o] = []
    
    for k in forest:
        cur_tree = forest[k]
        cur_depth = cur_tree.max_depth()
        cur_min = cur_tree.min_depth()
        cur_nodes = cur_tree.num_nodes()
        stat_dct['Key'].append(k)
        stat_dct['Max Depth'].append(cur_depth)
        stat_dct['Min Depth'].append(cur_min)
        stat_dct['Nodes'].append(cur_nodes)
    
    stat_df = pd.DataFrame(stat_dct)
    stat_df.to_csv('amazon_tree_stats.csv', index=False)
    
    

if __name__=='__main__':
    #get all the person connections from raw data
    # amazon_fpath = os.path.expanduser('~/amazon_roles/kaggle/test.csv')
    # get_person_con(amazon_fpath)
    
    #check if there are paths of length greater than 2
    # analyze_hierarchy('amazon_raw_userhierarchy.json')
    
    #extract the actual person hierarchies
    # if os.path.exists('amazon_spanningforest.pkl'):
    #     with open('amazon_spanningforest.pkl', 'rb') as fh:
    #         spanning_forest = pickle.load(fh)
    # else:
    #     spanning_forest = extract_hierarchy('amazon_raw_userhierarchy.json')
    #     #dump as a pkl
    #     with open('amazon_spanningforest.pkl', 'wb') as fh:
    #         pickle.dump(spanning_forest, fh)
    
    # print("Length of Spanning forest: {}".format(len(spanning_forest)))
    # print("Tree Depths:")
    # for tr_ind in spanning_forest:
    #     tr = spanning_forest[tr_ind]
    #     print(tr.max_depth())
    
    #and just make sure we can actually load the object back
    with open('amazon_spanningforest.pkl', 'rb') as fh:
        spanning_forest2 = pickle.load(fh)
    
    print_tree_stats(spanning_forest2)
    
    
    
    
    
    
    

