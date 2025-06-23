from models import *
from data_processing import *

def prepare_data_fold(source_pos_graph, source_neg_graph, num_folds=5):

    # Get the number of positive edges in the source graph
    num_pos_edges = source_pos_graph.number_of_edges(etype=('drug', 'DCpos', 'cell'))
    # Get the number of negative edges in the source graph
    num_neg_edges = source_neg_graph.number_of_edges(etype=('drug', 'DCneg', 'cell'))
    # Calculate the total number of edges
    num_edges = num_pos_edges + num_neg_edges
    # Generate indices for the edges
    idx = np.arange(num_edges)
    # Shuffle the indices randomly
    np.random.shuffle(idx)
    # Split the indices into 'num_folds' parts
    folds = np.array_split(idx, num_folds)

    return folds


def prepare_seed(seed_number=10):
    
    seed = seed_number    ### Seed for randomness
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def prepare_model():
    
    max_edges_per_type = 2000  # Set the maximum number of edges per type as needed
    
    # Initialize the model with input feature dimensions
    in_feats = {'drug': 167, 'cell': 5000, 'target': 147}
    # hidden_feats = 64
    # out_feats = 64
    
    rel_names2 = [('drug', 'DT', 'target'),('target', 'TD', 'drug'),('cell', 'CT', 'target'), ('target', 'TC', 'cell'), ('cell', 'CC1', 'cell'),('cell', 'CC2', 'cell'), ('target', 'TT1', 'target'), ('target', 'TT2', 'target')]
    
    # Initialize HeteroRGCN model
    gcn_model = HeteroRGCN(in_feats, 64, 64, rel_names2)
    
    # Initialize feature aligners
    feature_aligners = FeatureAlignAutoEncoder(64, 32)
    
    # Initialize edge aligners
    edge_aligners = EdgeAlignAutoEncoder(64 * 2, 32)
    
    # Initialize structure aligners
    structure_aligners = StructureAlignAutoEncoder(32 * 3, 32)
    
    # Initialize link predictor model
    link_predictor = HeteroDotProductPredictor()
    
    # Initialize MLPPredictor
    predictor = HeteroMLPPredictor(32, 1)
    
    # Set optimizer with learning rate
    optimizer = torch.optim.Adam(
        list(gcn_model.parameters()) + 
        list(feature_aligners.parameters()) +
        list(edge_aligners.parameters()) +
        list(structure_aligners.parameters()) +
        list(predictor.parameters()), lr=0.001)
    
    # Define loss functions
    criterion = nn.MSELoss()
    criterion_edge = torch.nn.BCEWithLogitsLoss()
    
    return gcn_model, feature_aligners, edge_aligners, structure_aligners, link_predictor, predictor, optimizer, criterion, criterion_edge, max_edges_per_type
