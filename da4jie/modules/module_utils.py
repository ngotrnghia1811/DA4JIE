import torch
import numpy as np

from collections import defaultdict


### SECTION: CRF_UTILS
### SECTION: 

def log_sum_exp(tensor, dim=0, keepdim: bool = False):
    """LogSumExp operation used by CRF."""
    m, _ = tensor.max(dim, keepdim=keepdim)
    if keepdim:
        stable_vec = tensor - m
    else:
        stable_vec = tensor - m.unsqueeze(dim)
    return m + (stable_vec.exp().sum(dim, keepdim=keepdim)).log()


def sequence_mask(lens, max_len=None):
    """Generate a sequence mask tensor from sequence lengths, used by CRF."""
    batch_size = lens.size(0)
    if max_len is None:
        max_len = lens.max().item()
    ranges = torch.arange(0, max_len, device=lens.device).long()
    ranges = ranges.unsqueeze(0).expand(batch_size, max_len)
    lens_exp = lens.unsqueeze(1).expand_as(ranges)
    mask = ranges < lens_exp
    return mask


### SECTION: GCN_UTILS
### SECTION: 

class Tree(object):
    """
    Reused tree object from stanfordnlp/treelstm.
    """

    def __init__(self):
        self.parent = None
        self.num_children = 0
        self.children = list()

    def add_child(self, child):
        child.parent = self
        self.num_children += 1
        self.children.append(child)

    def size(self):
        if getattr(self, '_size'):
            return self._size
        count = 1
        for i in range(self.num_children):
            count += self.children[i].size()
        self._size = count
        return self._size

    def depth(self):
        if getattr(self, '_depth'):
            return self._depth
        count = 0
        if self.num_children > 0:
            for i in range(self.num_children):
                child_depth = self.children[i].depth()
                if child_depth > count:
                    count = child_depth
            count += 1
        self._depth = count
        return self._depth

    def __iter__(self):
        yield self
        for c in self.children:
            for x in c:
                yield x


def get_head_token(head_id):
    return head_id - 1


def head_to_tree(head, actual_len, prune, subj_pos, obj_pos):
    """
    Convert a sequence of head indexes into a tree object.
    """
    head = head[:actual_len].tolist()
    root = None

    if prune < 0:
        nodes = [Tree() for _ in head]

        for i in range(len(nodes)):
            h = head[i]
            nodes[i].idx = i
            nodes[i].dist = -1  # just a filler
            if h == 0:
                root = nodes[i]
            else:
                nodes[get_head_token(head_id=h)].add_child(nodes[i])
    else:
        # find dependency path
        subj_pos = [i for i in range(actual_len) if subj_pos[i] == 0]
        obj_pos = [i for i in range(actual_len) if obj_pos[i] == 0]
        # with open('debug2.txt', 'w') as f:
        #     f.write('subj: {}'.format(subj_pos))
        #     f.write('\nobj: {}'.format(obj_pos))
        #     f.write('\nhead: {}'.format(head))
        common_ancestors = None

        subj_ancestors = set(subj_pos)
        for s in subj_pos:
            h = head[s]
            tmp = [s]
            while h > 0:
                head_token = get_head_token(head_id=h)
                tmp += [head_token]
                subj_ancestors.add(head_token)
                h = head[head_token]  # get parent node

            if common_ancestors is None:
                common_ancestors = set(tmp)
            else:
                common_ancestors.intersection_update(tmp)  # apply join operation between two sets

        obj_ancestors = set(obj_pos)
        for o in obj_pos:
            h = head[o]
            tmp = [o]
            while h > 0:
                head_token = get_head_token(head_id=h)
                tmp += [head_token]
                obj_ancestors.add(head_token)
                h = head[head_token]
            common_ancestors.intersection_update(tmp)

        # find lowest common ancestor
        if len(common_ancestors) == 1:
            lca = list(common_ancestors)[0]
        else:
            child_count = {k: 0 for k in common_ancestors}
            for ca in common_ancestors:
                head_of_ca = head[ca]
                head_token_of_ca = get_head_token(head_id=head_of_ca)
                if head_of_ca > 0 and head_token_of_ca in common_ancestors:
                    child_count[head_token_of_ca] += 1

            # the LCA has no child in the CA set
            for ca in common_ancestors:
                if child_count[ca] == 0:
                    lca = ca
                    break

        path_nodes = subj_ancestors.union(obj_ancestors).difference(common_ancestors)
        path_nodes.add(lca)

        # compute distance to path_nodes
        dist = [-1 if i not in path_nodes else 0 for i in range(actual_len)]

        for i in range(actual_len):
            if dist[i] < 0:
                stack = [i]
                while stack[-1] >= 0 and stack[-1] not in path_nodes:
                    stack.append(get_head_token(head[stack[-1]]))

                if stack[-1] in path_nodes:
                    for d, j in enumerate(reversed(stack)):
                        dist[j] = d
                else:
                    for j in stack:
                        if j >= 0 and dist[j] < 0:
                            dist[j] = int(1e4)  # aka infinity

        highest_node = lca
        nodes = [Tree() if dist[i] <= prune else None for i in range(actual_len)]

        for i in range(len(nodes)):
            if nodes[i] is None:
                continue
            h = head[i]
            nodes[i].idx = i
            nodes[i].dist = dist[i]
            if h > 0 and i != highest_node:
                head_token = get_head_token(head_id=h)
                assert nodes[head_token] is not None
                nodes[head_token].add_child(nodes[i])

        root = nodes[highest_node]

    assert root is not None
    return root


def tree_to_adj(sent_len, tree, directed=True, self_loop=False):
    """
    Convert a tree object to an (numpy) adjacency matrix.
    """
    ret = np.zeros((sent_len, sent_len), dtype=np.float32)

    queue = [tree]
    idx = []
    while len(queue) > 0:
        t, queue = queue[0], queue[1:]

        idx += [t.idx]

        for c in t.children:
            ret[t.idx, c.idx] = 1
        queue += t.children

    if not directed:
        ret = ret + ret.T

    if self_loop:
        for i in idx:
            ret[i, i] = 1

    return ret


def tree_to_dist(sent_len, tree):
    ret = -1 * np.ones(sent_len, dtype=np.int64)

    for node in tree:
        ret[node.idx] = node.dist

    return ret


def get_full_adj(head_ids, self_loop=False):
    with torch.no_grad():
        # head_ids = head_ids.data.cpu().numpy()
        batch_size, seq_len = head_ids.shape
        adj_list = []
        for b_id in range(batch_size):
            adj = np.zeros((seq_len, seq_len))
            for i in range(seq_len):
                adj[i][i] = 1 if self_loop else 0
                head = int(head_ids[b_id][i] - 1)
                if head >= 0:
                    adj[i][head] = 1
                    adj[head][i] = 1
            adj_list.append(adj)
        adj_list = np.array(adj_list)
        adj_maxtrix = head_ids.new_tensor(adj_list, requires_grad=False)
        return adj_maxtrix.float()


def get_pruned_adj(head_ids, subj_positions, obj_positions, pad_masks,
                   prune):
    '''
    subj_positions, obj_positions: 0 at positions of subj, obj, otherwise, !=0
    subj_positions: [num words, ]
    obj_positions: [num words, ]
    head_ids: [num words, ]
    '''
    with torch.no_grad():
        lengths = (pad_masks.data.cpu().numpy() == 0).astype(np.int64).sum(
            1)  # [batch size, ] actual length of each sequence in the batch
        maxlen = max(lengths)
        head_ids, subj_positions, obj_positions = head_ids.cpu().numpy(), subj_positions.cpu().numpy(), obj_positions.cpu().numpy()
        trees = [head_to_tree(head_ids[i], lengths[i], prune, subj_positions[i],
                              obj_positions[i]) for i in range(len(lengths))]
        adj = [tree_to_adj(maxlen, tree, directed=False, self_loop=False).reshape(1, maxlen, maxlen) for tree in
               trees]
        adj = np.concatenate(adj, axis=0)
        adj = torch.from_numpy(adj)  # [batch size, max len, max len]
        adj = adj.float().cuda()
    return adj


### SECTION: 
### SECTION: 