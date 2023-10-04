import json
from torch_geometric.data import HeteroData
import torch

'''

DATASET FORMAT
------------------------------------

abstract_text: string (abstract)

referenced_works: list of strings (urls)

publication_date: string (date)

display_name: string (title)

concepts: dict
    id: string (url)
    wikidata: string (url)
    display_name: string
    level: int
    score: float

has_fulltext: bool (NOT USED IN DATA PROCESSING)

best_oa_location: string containing dict? (NOT USED IN DATA PROCESSING)

'''

# returns a HeteroData object, index to paper/concept mappings, non-numerical metadata
# HeteroData allows for multiple node types (in this case, paper and concept)
# HeteroData can be passed into GNN for training

def load_data(data_name):
    # load json dataset
    with open(data_name, 'r') as f:
        data = json.load(f)

    # assign each paper and concept an integer
    paper_to_index = dict()
    concept_to_index = dict()
    papers = []
    concepts = []
    for paper in data:
        if paper not in papers:
            paper_to_index[paper] = len(paper_to_index)
            papers.append(paper)
        for concept in data[paper]["concepts"]:
            if concept["id"] not in concept_to_index:
                concept_to_index[concept["id"]] = len(concept_to_index)
                concepts.append(concept)

    # node feature matrices for each node type
    # no numerical data to encode for papers or concepts
    node_features = {
        'paper' : torch.ones(len(papers), 1),
        'concept' : torch.ones(len(concepts), 1)
    }

    # edge processing for paper -> concept edge type
    source_pc = []
    target_pc = []
    attrs_pc = []
    for paper in data:
        for concept in data[paper]["concepts"]:
            source_pc.append(paper_to_index[paper])
            target_pc.append(concept_to_index[concept["id"]])
            attrs_pc.append([concept["level"], concept["score"]])
    
    # edge processing for paper -> paper edge type
    source_pp = []
    target_pp = []
    for paper in data:
        for ref in data[paper]["referenced_works"]:
            source_pp.append(paper_to_index[paper])
            target_pp.append(paper_to_index[ref])

    # edge index tensors for each edge type
    edge_index = {
        ('paper', 'contains', 'concept') : torch.tensor([source_pc, target_pc]),
        ('paper', 'references', 'paper') : torch.tensor([source_pp, target_pp])
    }

    # edge feature matrices for each edge type
    edge_attr = {
        ('paper', 'contains', 'concept') : torch.tensor(attrs_pc)
    }

    # instantiate the HeteroData object
    graph_data = HeteroData(
        node_dict = node_features,
        edge_dict = edge_index,
        edge_attr_dict = edge_attr
    )

    # graph_data is the HeteroData object that should be passed into a GNN
    return graph_data, papers, concepts, data

# graph_data, papers, concepts, data = load_data('10_000-MIT-AI-papers.json')
