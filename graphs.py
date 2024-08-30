from typing import List, Tuple, Dict

import networkx as nx

TRIPLE_VERB_TEMPLATE = "({head},{relation},{tail})\n"


def get_nx_graph(triples: List[Tuple[str, str, str]], concept_2_id, relation_2_id) -> nx.DiGraph:
    """
    Create a NetworkX DiGraph from a list of triples.
    """
    graph = nx.DiGraph()
    for triple in triples:
        graph.add_edge(concept_2_id[triple[0]], concept_2_id[triple[2]],
                       relation=relation_2_id[triple[1]])
    return graph


def get_neighbors(graph: nx.DiGraph, concept: str, concept_2_id: Dict[str, int],
                  id_2_concept: Dict[int, str], mode='bidirectional') -> List[str]:
    # Get the concept ID from its name
    if concept not in concept_2_id:
        return []

    concept_id = concept_2_id[concept]
    if concept_id not in graph:
        return []

    if mode == 'bidirectional':
        # Get all neighbors (successors and predecessors)
        neighbors = set(graph.predecessors(concept_id)).union(set(graph.successors(concept_id)))
    elif mode == 'outgoing':
        neighbors = set(graph.successors(concept_id))
    elif mode == 'ingoing':
        neighbors = set(graph.predecessors(concept_id))
    else:
        raise ValueError(f"Invalid mode: {mode}")

    # Return the names of the neighbors
    neighbor_list = [id_2_concept[id] for id in neighbors]
    return neighbor_list


def get_2hop_neighbors(graph: nx.DiGraph, concept: str, concept_2_id: Dict[str, int],
                       id_2_concept: Dict[int, str]) -> List[str]:
    # Get the concept ID from its name
    if concept not in concept_2_id:
        return []
    concept_id = concept_2_id[concept]
    if concept_id not in graph:
        return []

    # Get 1-hop neighbors (successors and predecessors)
    neighbors_1hop = set(get_neighbors(graph, concept, concept_2_id, id_2_concept))

    # Initialize a set for 2-hop neighbors
    neighbors_2hop = set()

    # Find 2-hop neighbors by looking at neighbors of 1-hop neighbors
    for neighbor in neighbors_1hop:
        neighbors_2hop.update(set(graph.predecessors(concept_2_id[neighbor])))
        neighbors_2hop.update(set(graph.successors(concept_2_id[neighbor])))

    # Remove the original concept and 1-hop neighbors from the 2-hop neighbors set
    neighbors_2hop.discard(concept_id)
    neighbors_2hop -= neighbors_1hop

    # Return the names of the 2-hop neighbors
    neighbor_list = [id_2_concept[id] for id in neighbors_2hop]
    return neighbor_list


def verbalize_neighbors_triples_from_graph(graph: nx.DiGraph, concept, concept_2_id, id_2_concept,
                                           unifyed_relation='Is-a-Prerequisite-of', mode='outgoing') -> str:
    neighbors = get_neighbors(graph, concept, concept_2_id, id_2_concept, mode=mode)
    if len(neighbors) == 0:
        return "None"
    res_str = ''
    for neighbor in neighbors:
        res_str += TRIPLE_VERB_TEMPLATE.format(head=concept, relation=unifyed_relation,
                                               tail=neighbor)

    return res_str


def verbalize_neighbors_triples_from_triples(graph: List[Tuple[str, str, str]], concept: str) -> str:
    """
    Get the triples that connect a concept to its neighbors in the graph.
    """
    if len(graph) == 0:
        return "None"
    res_str = ''
    for triple in graph:
        if concept == triple[0] or concept == triple[2]:
            res_str += TRIPLE_VERB_TEMPLATE.format(head=triple[0], relation=triple[1],
                                                   tail=triple[2])
    return res_str
