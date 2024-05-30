import numpy as np

class Graph():
    '''The class to generate the graph of the given joints'''
    def __init__(self,
                 max_hop=1,
                 strategy='uniform',
                 dilation=1):
        self.max_hop = max_hop
        self.dilation = dilation

        self.get_edge()
        self.hop_dis = self.get_hop_distance(
            self.num_node, self.edge, max_hop=max_hop)
        self.get_adjacency(strategy)



    def get_edge(self):
        self.num_node = 22
        self_link = [(i, i) for i in range(self.num_node)]
        neighbor_link = [(11, 12), (11, 13), (11, 23), (12, 14), (12, 24),
                         (13, 15), (14, 16), (15, 17), (15, 19), (15, 21),
                         (16, 18), (16, 20), (16, 22), (17, 19), (18, 20),
                         (23, 24), (23, 25), (24, 26), (25, 27), (26, 28),
                         (27, 29), (27, 31), (28, 30), (28, 32), (29, 31),
                         (30, 32)]
        # subtract 11 for neighbor_link because the index of joints in the dataset starts from 0
        neighbor_link = [(i - 11, j - 11) for (i, j) in neighbor_link]
        self.edge = self_link + neighbor_link
        self.center = 1

    def get_adjacency(self, strategy):
        valid_hop = range(0, self.max_hop + 1, self.dilation)

        adjacency = np.zeros((self.num_node, self.num_node))
        for hop in valid_hop:
            adjacency[self.hop_dis == hop] = 1
        normalize_adjacency = self.normalize_digraph(adjacency)

        if strategy == 'uniform':
            A = np.zeros((1, self.num_node, self.num_node))
            A[0] = normalize_adjacency
            self.A = A
        elif strategy == 'distance':
            A = np.zeros((len(valid_hop), self.num_node, self.num_node))
            for i, hop in enumerate(valid_hop):
                A[i][self.hop_dis == hop] = normalize_adjacency[self.hop_dis ==
                                                                hop]
            self.A = A
        elif strategy == 'spatial':
            A = []
            for hop in valid_hop:
                a_root = np.zeros((self.num_node, self.num_node))
                a_close = np.zeros((self.num_node, self.num_node))
                a_further = np.zeros((self.num_node, self.num_node))
                for i in range(self.num_node):
                    for j in range(self.num_node):
                        if self.hop_dis[j, i] == hop:
                            if self.hop_dis[j, self.center] == self.hop_dis[
                                    i, self.center]:
                                a_root[j, i] = normalize_adjacency[j, i]
                            elif self.hop_dis[j, self.
                                              center] > self.hop_dis[i, self.
                                                                     center]:
                                a_close[j, i] = normalize_adjacency[j, i]
                            else:
                                a_further[j, i] = normalize_adjacency[j, i]
                if hop == 0:
                    A.append(a_root)
                else:
                    A.append(a_root + a_close)
                    A.append(a_further)
            A = np.stack(A)
            self.A = A
        else:
            raise ValueError("Do Not Exist This Strategy")

    def get_hop_distance(self, num_node, edge, max_hop=1):
        A = np.zeros((num_node, num_node))
        for i, j in edge:
            A[j, i] = 1
            A[i, j] = 1

        # compute hop steps
        hop_dis = np.zeros((num_node, num_node)) + np.inf
        transfer_mat = [np.linalg.matrix_power(A, d) for d in range(max_hop + 1)]
        arrive_mat = (np.stack(transfer_mat) > 0)
        for d in range(max_hop, -1, -1):
            hop_dis[arrive_mat[d]] = d
        return hop_dis


    def normalize_digraph(self,A):
        Dl = np.sum(A, 0)
        num_node = A.shape[0]
        Dn = np.zeros((num_node, num_node))
        for i in range(num_node):
            if Dl[i] > 0:
                Dn[i, i] = Dl[i]**(-1)
        AD = np.dot(A, Dn)
        return AD


    def normalize_undigraph(self, A):
        Dl = np.sum(A, 0)
        num_node = A.shape[0]
        Dn = np.zeros((num_node, num_node))
        for i in range(num_node):
            if Dl[i] > 0:
                Dn[i, i] = Dl[i]**(-0.5)
        DAD = np.dot(np.dot(Dn, A), Dn)
        return DAD


