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
        pdb_path_gz = "{}/{}/{}.pdb.gz".format(args.data_path, args.pdb_path, protein)
        pdb_path = "{}/{}/{}.pdb".format(args.data_path, args.pdb_path, protein)
        
        if os.path.exists(pdb_path_gz):
            alpha_fold_seq = get_sequence_from_pdb(pdb_path_gz, "A")
        elif os.path.exists(pdb_path):
            alpha_fold_seq = get_sequence_from_pdb(pdb_path, "A")
        else:
            print(f"Warning: PDB file not found for protein {protein} (tried both .pdb and .pdb.gz)")
            continue
            
        fasta.append(create_seqrecord(id=protein, seq=alpha_fold_seq))
    
    SeqIO.write(fasta, "{}/sequence.fasta".format(args.data_path), "fasta")
    args.fasta_path = "{}/sequence.fasta".format(args.data_path)


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
        embedding_path = "{}/esm/{}.pt".format(args.data_path, protein)
        if os.path.exists(embedding_path):
            existing_embeddings.append(protein)
        else:
            missing_embeddings.append(protein)
    
    print(f"Found embeddings for {len(existing_embeddings)} proteins")
    if missing_embeddings:
        print(f"Missing embeddings for {len(missing_embeddings)} proteins: {missing_embeddings[:5]}{'...' if len(missing_embeddings) > 5 else ''}")
    
    return existing_embeddings, missing_embeddings


# Initialize proteins list
proteins = []

if args.input_type == 'fasta':
    if args.fasta_path:
        fasta_full_path = "{}/{}".format(args.data_path, args.fasta_path)
        if os.path.exists(fasta_full_path):
            proteins = set(get_proteins_from_fasta(fasta_full_path))
            print(f"Found {len(proteins)} proteins in FASTA file")
            
            # Get all PDB files (both .pdb and .pdb.gz)
            pdb_dir = "{}/{}".format(args.data_path, args.pdb_path)
            if os.path.exists(pdb_dir):
                pdb_files = os.listdir(pdb_dir)
                pdbs = set()
                for pdb_file in pdb_files:
                    if pdb_file.endswith(".pdb.gz"):
                        pdbs.add(pdb_file.replace(".pdb.gz", ""))
                    elif pdb_file.endswith(".pdb"):
                        pdbs.add(pdb_file.replace(".pdb", ""))
                
                print(f"Found {len(pdbs)} PDB files")
                proteins = list(pdbs.intersection(proteins))
                print(f"Intersection: {len(proteins)} proteins have both FASTA and PDB")
            else:
                print(f"PDB directory not found: {pdb_dir}")
                exit()
        else:
            print(f"FASTA file not found: {fasta_full_path}")
            exit()
        
elif args.input_type == 'pdb':
    pdb_path = "{}/{}".format(args.data_path, args.pdb_path)
    if os.path.exists(pdb_path):
        pdb_files = os.listdir(pdb_path)
        proteins = []
        for pdb_file in pdb_files:
            if pdb_file.endswith(".pdb.gz"):
                proteins.append(pdb_file.replace(".pdb.gz", ""))
            elif pdb_file.endswith(".pdb"):
                proteins.append(pdb_file.replace(".pdb", ""))
        
        if len(proteins) == 0:
            print("No proteins found in {}.".format(pdb_path))
            exit()
        create_fasta(proteins)
    else:
        print("PDB directory not found -- {}".format(pdb_path))
        exit()

if len(proteins) > 0:
    print("Found {} total proteins".format(len(proteins)))
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
else:
    # Generate embeddings (original code)
    print("Generating Embeddings from {}".format(args.fasta_path))
    os.makedirs("{}/esm".format(args.data_path), exist_ok=True)
    generate_embeddings(args.fasta_path)

print(f"Proceeding with prediction for {len(proteins)} proteins")

# Create dataset kwargs
kwargs = {
    'seq_id': Constants.Final_thresholds[args.ontology],
    'ont': args.ontology,
    'session': 'selected',
    'prot_ids': proteins,
    'pdb_path': "{}/{}".format(args.data_path, args.pdb_path)
}

print("Loading dataset...")
dataset = load_dataset(root=args.data_path, **kwargs)

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

ckp_pth = "{}/{}.pt".format(args.data_path, args.ontology)
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
goterms = pickle_load('{}/go_terms'.format(args.data_path))[f'GO-terms-{args.ontology}']
go_graph = obonet.read_obo(open("{}/go-basic.obo".format(args.data_path), 'r'))
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
output_path = "{}/{}".format(args.data_path, args.output)
write_to_file(results, output_path)

print(f"Prediction complete!")
print(f"Results saved to: {output_path}")
print(f"Predicted functions for {len(results)} proteins")

# Print summary statistics
total_predictions = sum(len(protein_scores) for protein_scores in results.values())
avg_predictions_per_protein = total_predictions / len(results) if results else 0
print(f"Total GO term predictions: {total_predictions}")
print(f"Average predictions per protein: {avg_predictions_per_protein:.2f}")