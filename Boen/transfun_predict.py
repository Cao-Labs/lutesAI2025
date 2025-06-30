import argparse
import os

import networkx as nx
import obonet
import torch
from Bio import SeqIO
from torch import optim
from torch_geometric.loader import DataLoader
import Constants
import params
from Dataset.Dataset import load_dataset
from models.gnn import GCN
from preprocessing.utils import load_ckp, get_sequence_from_pdb, create_seqrecord, get_proteins_from_fasta, \
    generate_bulk_embedding, pickle_load, fasta_to_dictionary

parser = argparse.ArgumentParser(description=" Predict protein functions with TransFun ", epilog=" Thank you !!!")
parser.add_argument('--data-path', type=str, default="/data/summer2020/Boen/TransFun/data", help="Path to data files")
parser.add_argument('--ontology', type=str, default="cellular_component", help="Ontology to predict")
parser.add_argument('--no-cuda', default=False, help='Disables CUDA training.')
parser.add_argument('--batch-size', default=10, help='Batch size.')
parser.add_argument('--input-type', choices=['fasta', 'pdb'], default="fasta",
                    help='Input Data: fasta file or PDB files')
parser.add_argument('--fasta-path', default="benchmark_testing_sequences.fasta", help='Path to Fasta')
parser.add_argument('--pdb-path', default="benchmark_testing_pdbs_renamed", help='Path to directory of PDBs')
parser.add_argument('--cut-off', type=float, default=0.0, help="Cut of to report function")
parser.add_argument('--output', type=str, default="output", help="File to save output")
parser.add_argument('--skip-embedding', action='store_true', default=True, help="Skip embedding generation (embeddings already exist)")

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

if args.cuda:
    device = 'cuda'
else:
    device = 'cpu'

if args.ontology == 'molecular_function':
    ont_kwargs = params.mol_kwargs
elif args.ontology == 'cellular_component':
    ont_kwargs = params.cc_kwargs
elif args.ontology == 'biological_process':
    ont_kwargs = params.bio_kwargs
ont_kwargs['device'] = device

FUNC_DICT = {
    'cellular_component': 'GO:0005575',
    'molecular_function': 'GO:0003674',
    'biological_process': 'GO:0008150'
}

print("Predicting proteins (skipping embedding generation)")

def create_fasta(proteins):
    fasta = []
    for protein in proteins:
        # Try both .pdb.gz and .pdb extensions
        pdb_path_gz = os.path.join(args.data_path, args.pdb_path, f"{protein}.pdb.gz")
        pdb_path = os.path.join(args.data_path, args.pdb_path, f"{protein}.pdb")
        
        if os.path.exists(pdb_path_gz):
            alpha_fold_seq = get_sequence_from_pdb(pdb_path_gz, "A")
        elif os.path.exists(pdb_path):
            alpha_fold_seq = get_sequence_from_pdb(pdb_path, "A")
        else:
            print(f"Warning: PDB file not found for protein {protein} (tried both .pdb and .pdb.gz)")
            continue
            
        fasta.append(create_seqrecord(id=protein, seq=alpha_fold_seq))
    
    SeqIO.write(fasta, os.path.join(args.data_path, "sequence.fasta"), "fasta")
    args.fasta_path = os.path.join(args.data_path, "sequence.fasta")


def write_to_file(data, output):
    with open('{}'.format(output), 'w') as fp:
        for protein, go_terms in data.items():
            for go_term, score in go_terms.items():
                fp.write('%s %s %s\n' % (protein, go_term, score))


def check_embeddings_exist(proteins):
    """Check if embeddings already exist for all proteins"""
    missing_embeddings = []
    existing_embeddings = []
    
    for protein in proteins:
        embedding_path = os.path.join(args.data_path, "esm", f"{protein}.pt")
        if os.path.exists(embedding_path):
            existing_embeddings.append(protein)
        else:
            missing_embeddings.append(protein)
    
    print(f"Found embeddings for {len(existing_embeddings)} proteins")
    if missing_embeddings:
        print(f"Missing embeddings for {len(missing_embeddings)} proteins: {missing_embeddings[:5]}{'...' if len(missing_embeddings) > 5 else ''}")
    
    return existing_embeddings, missing_embeddings


def check_pdb_files_exist(proteins):
    """Check which proteins have PDB files and return only those that exist"""
    existing_pdbs = []
    missing_pdbs = []
    
    pdb_dir = os.path.join(args.data_path, args.pdb_path)
    
    for protein in proteins:
        pdb_path_gz = os.path.join(pdb_dir, f"{protein}.pdb.gz")
        pdb_path = os.path.join(pdb_dir, f"{protein}.pdb")
        
        if os.path.exists(pdb_path_gz) or os.path.exists(pdb_path):
            existing_pdbs.append(protein)
        else:
            missing_pdbs.append(protein)
    
    print(f"Found PDB files for {len(existing_pdbs)} proteins")
    if missing_pdbs:
        print(f"Missing PDB files for {len(missing_pdbs)} proteins: {missing_pdbs[:5]}{'...' if len(missing_pdbs) > 5 else ''}")
    
    return existing_pdbs, missing_pdbs


# Initialize proteins list
proteins = []

if args.input_type == 'fasta':
    if args.fasta_path:
        fasta_full_path = os.path.join(args.data_path, args.fasta_path)
        if os.path.exists(fasta_full_path):
            proteins = set(get_proteins_from_fasta(fasta_full_path))
            print(f"Found {len(proteins)} proteins in FASTA file")
            
            # Check which proteins have PDB files
            existing_pdbs, missing_pdbs = check_pdb_files_exist(proteins)
            proteins = existing_pdbs
            
            if len(proteins) == 0:
                print("No proteins with PDB files found!")
                exit()
        else:
            print(f"FASTA file not found: {fasta_full_path}")
            exit()
        
elif args.input_type == 'pdb':
    pdb_dir = os.path.join(args.data_path, args.pdb_path)
    if os.path.exists(pdb_dir):
        pdb_files = os.listdir(pdb_dir)
        proteins = []
        for pdb_file in pdb_files:
            if pdb_file.endswith(".pdb.gz"):
                proteins.append(pdb_file.replace(".pdb.gz", ""))
            elif pdb_file.endswith(".pdb"):
                proteins.append(pdb_file.replace(".pdb", ""))
        
        if len(proteins) == 0:
            print("No proteins found in {}.".format(pdb_dir))
            exit()
        create_fasta(proteins)
    else:
        print("PDB directory not found -- {}".format(pdb_dir))
        exit()

if len(proteins) > 0:
    print("Found {} total proteins with PDB files".format(len(proteins)))
else:
    print("No proteins found for prediction.")
    exit()

# Check which proteins have embeddings
if args.skip_embedding:
    existing_proteins, missing_proteins = check_embeddings_exist(proteins)
    
    if missing_proteins:
        print(f"WARNING: {len(missing_proteins)} proteins are missing embeddings!")
        print("You may need to generate embeddings for these proteins first.")
        
        # Use only proteins with existing embeddings
        proteins = existing_proteins
        print(f"Proceeding with {len(proteins)} proteins that have embeddings")
    
    if len(proteins) == 0:
        print("No proteins with embeddings found. Cannot proceed.")
        exit()

print(f"Final protein list: {len(proteins)} proteins")

# Ensure PDB path has proper format for dataset loading
# Fix the path concatenation bug by ensuring proper path joining
pdb_path_for_dataset = os.path.join(args.data_path, args.pdb_path)

# Create dataset kwargs with proper path handling
kwargs = {
    'seq_id': Constants.Final_thresholds[args.ontology],
    'ont': args.ontology,
    'session': 'selected',
    'prot_ids': proteins,
    'pdb_path': pdb_path_for_dataset
}

print(f"Dataset kwargs:")
print(f"  - seq_id: {kwargs['seq_id']}")
print(f"  - ontology: {kwargs['ont']}")
print(f"  - session: {kwargs['session']}")
print(f"  - proteins: {len(kwargs['prot_ids'])}")
print(f"  - pdb_path: {kwargs['pdb_path']}")

print("Loading dataset...")
try:
    dataset = load_dataset(root=args.data_path, **kwargs)
    print(f"Successfully loaded dataset with {len(dataset)} samples")
except Exception as e:
    print(f"Error loading dataset: {e}")
    print("Debugging information:")
    print(f"  - Data path: {args.data_path}")
    print(f"  - PDB path: {pdb_path_for_dataset}")
    print(f"  - First 5 proteins: {proteins[:5]}")
    
    # Check if PDB files exist for first few proteins
    for protein in proteins[:3]:
        pdb_gz = os.path.join(pdb_path_for_dataset, f"{protein}.pdb.gz")
        pdb_regular = os.path.join(pdb_path_for_dataset, f"{protein}.pdb")
        print(f"  - {protein}: .pdb.gz exists: {os.path.exists(pdb_gz)}, .pdb exists: {os.path.exists(pdb_regular)}")
    
    raise e

print("Creating data loader...")
test_dataloader = DataLoader(dataset,
                             batch_size=args.batch_size,
                             drop_last=False,
                             shuffle=False)

# model
print("Loading model...")
model = GCN(**ont_kwargs)
model.to(device)

optimizer = optim.Adam(model.parameters())

ckp_pth = os.path.join(args.data_path, f"{args.ontology}.pt")
print(f"Loading checkpoint from: {ckp_pth}")

# load the saved checkpoint
if os.path.exists(ckp_pth):
    model, optimizer, current_epoch, min_val_loss = load_ckp(ckp_pth, model, optimizer, device)
    print(f"Loaded model from epoch {current_epoch}")
else:
    print("Model not found. Skipping...")
    exit()

print("Running predictions...")
model.eval()

scores = []
protein_results = []

for batch_idx, data in enumerate(test_dataloader):
    print(f"Processing batch {batch_idx + 1}/{len(test_dataloader)}")
    with torch.no_grad():
        protein_results.extend(data['atoms'].protein)
        scores.extend(model(data.to(device)).tolist())

assert len(protein_results) == len(scores)
print(f"Generated predictions for {len(protein_results)} proteins")

print("Loading GO terms and ontology graph...")
goterms = pickle_load(os.path.join(args.data_path, 'go_terms'))[f'GO-terms-{args.ontology}']
go_graph = obonet.read_obo(open(os.path.join(args.data_path, "go-basic.obo"), 'r'))
go_set = nx.ancestors(go_graph, FUNC_DICT[args.ontology])

print("Processing predictions and applying hierarchical constraints...")
results = {}
for protein, score in zip(protein_results, scores):
    protein_scores = {}

    # Apply cutoff threshold
    for go_term, _score in zip(goterms, score):
        if _score > args.cut_off:
            protein_scores[go_term] = max(protein_scores.get(go_term, 0), _score)

    # Apply hierarchical constraints (propagate scores to descendants)
    for go_term, max_score in list(protein_scores.items()):
        descendants = nx.descendants(go_graph, go_term).intersection(go_set)
        for descendant in descendants:
            protein_scores[descendant] = max(protein_scores.get(descendant, 0), max_score)

    results[protein] = protein_scores

print("Writing output to {}".format(args.output))
output_path = os.path.join(args.data_path, args.output)
write_to_file(results, output_path)

print(f"Prediction complete!")
print(f"Results saved to: {output_path}")
print(f"Predicted functions for {len(results)} proteins")

# Print summary statistics
total_predictions = sum(len(protein_scores) for protein_scores in results.values())
avg_predictions_per_protein = total_predictions / len(results) if results else 0
print(f"Total GO term predictions: {total_predictions}")
print(f"Average predictions per protein: {avg_predictions_per_protein:.2f}")