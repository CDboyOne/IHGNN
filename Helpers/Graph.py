from typing import List, Dict, Set, Any, Tuple
from Helpers.Torches import *
from Helpers.SearchLog import PosInteraction
from Helpers.SearchLogCollection import SearchLogCollection
from Helpers.GlobalSettings import Gs, Gsv

class PpsGraph:
    '''图结构的基类。'''
    def __init__(self): pass


class Pps2DGraph(PpsGraph):

    Adjacency: Tensor     # 邻接矩阵，是稀疏矩阵
    VertexDegrees: Tensor # 结点度矩阵，行数为结点数量，列数为 1
    
    def __init__(self): super().__init__()

    @classmethod
    def from_interactions(cls, 
        interactions: List[PosInteraction], 
        node_count: int,
        user_count: int, 
        query_count: int,
        use_self_connection: bool,
        device: torch.device):

        if use_self_connection:
            indicesX, indicesY, elements = list(range(node_count)), list(range(node_count)), [1] * node_count
            vertex_degrees = torch.ones(node_count, dtype=torch.float, device=device)
        else:
            indicesX, indicesY, elements = [], [], []
            vertex_degrees = torch.zeros(node_count, dtype=torch.float, device=device)

        for pos_interaction in interactions:
            u, q, i, flag = pos_interaction.uqif()
            if flag > 0:
                q += user_count
                i += (query_count + user_count)
                if Gs.graph_completeness == Gsv.graph_uqi:
                    # u-i u-q q-i; i-u q-u i-q
                    indicesX.extend([u, q, i, i, q, u])
                    indicesY.extend([q, i, u, q, u, i])
                    elements.extend([1, 1, flag, 1, 1, flag])
                    vertex_degrees[[u, q, i]] += 2
                elif Gs.graph_completeness == Gsv.graph_only_uq:
                    # u-q; q-u
                    indicesX.extend([u, q])
                    indicesY.extend([q, u])
                    elements.extend([1, 1])
                    vertex_degrees[[u, q]] += 1
                elif Gs.graph_completeness == Gsv.graph_only_ui:
                    # u-i; i-u
                    indicesX.extend([u, i])
                    indicesY.extend([i, u])
                    elements.extend([1, 1])
                    vertex_degrees[[u, i]] += 1
                elif Gs.graph_completeness == Gsv.graph_only_qi:
                    # q-i; i-q
                    indicesX.extend([q, i])
                    indicesY.extend([i, q])
                    elements.extend([1, 1])
                    vertex_degrees[[q, i]] += 1
                else:
                    raise ValueError()
        
        if not use_self_connection:
            vertex_degrees[vertex_degrees == 0] = 1e-8
        
        g = Pps2DGraph()
        g.Adjacency = torch.sparse_coo_tensor(
            indices=[indicesX, indicesY],
            values=elements,
            size=(node_count, node_count),
            dtype=torch.float,
            device=device
        ).coalesce()
        
        # 要想让 Dv 作为结点度矩阵来左乘其他矩阵（Dv 放在左边），就要让它变成只有一列的二维矩阵
        g.VertexDegrees = vertex_degrees.view(-1, 1)
        return g


class PpsHyperGraph(PpsGraph):

    Adjacency: Tensor # 关联矩阵
    I3: LongTensor    # 交互矩阵，行数为交互数量，列数为 3（每行代表一个 uqi 交互）；在超图中，行索引即为边的编号
    VertexDegrees: Tensor
    EdgeDegrees: Tensor
    EdgeCount: int

    def __init__(self): super().__init__()

    @classmethod
    def from_interactions(cls,
        interactions: List[PosInteraction], 
        node_count: int,
        user_count: int, 
        query_count: int,
        device: torch.device):

        indicesX, indicesY, elements = [], [], []
        vertex_degrees = torch.zeros(node_count, dtype=torch.float, device=device)
        edge = 0
        i3 = []

        for pos_interaction in interactions:
            u, q, i, flag = pos_interaction.uqif()
            if flag > 0:
                q += user_count
                i += (query_count + user_count)
                vertex_degrees[[u, q, i]] += 1
                # u-edge, q-edge, i-edge
                indicesX.extend([u,    q,    i])
                indicesY.extend([edge, edge, edge])
                elements.extend([1] * 3)
                i3.append([u, q, i])
                edge += 1

        vertex_degrees[vertex_degrees == 0] = 1e-8
        
        g = PpsHyperGraph()
        g.Adjacency = torch.sparse_coo_tensor(
            [indicesX, indicesY], 
            elements, 
            [node_count, edge], 
            dtype=torch.float, device=device
        ).coalesce()
        g.I3 = torch.tensor(i3, dtype=torch.long, device=device)
        # 要想让 Dv 作为结点度矩阵来左乘其他矩阵（Dv 放在左边），就要让它变成只有一列的二维矩阵；De 同理
        g.VertexDegrees = vertex_degrees.view(-1, 1)
        g.EdgeDegrees = Tensor([3] * edge).to(device).view(-1, 1)
        g.EdgeCount = edge
        return g



class PpsLogHyperGraph(PpsGraph):
    
    Adjacency: Tensor # 关联矩阵
    I3: LongTensor    # 交互矩阵，行数为交互数量，列数为 3（每行代表一个 uqi 交互）；在超图中，行索引即为边的编号
    VertexDegrees: Tensor
    EdgeDegrees: Tensor
    EdgeCount: int

    def __init__(self): super().__init__()

    @classmethod
    def from_search_logs(cls,
        logs: SearchLogCollection,
        node_count: int,
        user_count: int, 
        query_count: int,
        device: torch.device):

        indicesX, indicesY, elements = [], [], []
        vertex_degrees = torch.zeros(node_count, dtype=torch.float, device=device)
        edge_degrees = []
        edge = 0

        for log in logs:
            u, q = log.user, log.query + user_count
            nodes = [u, q]
            nodes.extend([item + query_count + user_count for item, flag in zip(log.items, log.interactions) if flag > 0])
            if len(nodes) == 2: continue

            # u-edge, q-edge, i0-edge, i1-edge, ...
            vertex_degrees[nodes] += 1
            edge_degrees.append(len(nodes))
            indicesX.extend(nodes)
            indicesY.extend([edge] * len(nodes))
            elements.extend([1] * len(nodes))
            edge += 1
        
        vertex_degrees[vertex_degrees == 0] = 1e-8
        
        g = PpsLogHyperGraph()
        g.Adjacency = torch.sparse_coo_tensor(
            [indicesX, indicesY], 
            elements, 
            [node_count, edge], 
            dtype=torch.float, 
            device=device
        ).coalesce()
        # 要想让 Dv 作为结点度矩阵来左乘其他矩阵（Dv 放在左边），就要让它变成只有一列的二维矩阵；De 同理
        g.VertexDegrees = vertex_degrees.view(-1, 1)
        g.EdgeDegrees = Tensor(edge_degrees).to(device).view(-1, 1)
        g.EdgeCount = edge
        return g