import sys
import os
from GeneOntologyTree import GeneOntologyTree

def filter_ground_truth_by_ontology(obo_file, input_ground_truth, output_dir):
    """
    Filter the ground truth file into separate ontology-specific files.
    
    Args:
        obo_file: Path to the GO OBO file
        input_ground_truth: Path to the consolidated ground truth file
        output_dir: Directory to save filtered files
    """
    
    print("Loading Gene Ontology tree...")
    go_tree = GeneOntologyTree(obo_file, TestMode=0)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Output files for each ontology
    bp_file = os.path.join(output_dir, "ground_truth_bp.tsv")
    cc_file = os.path.join(output_dir, "ground_truth_cc.tsv") 
    mf_file = os.path.join(output_dir, "ground_truth_mf.tsv")
    
    bp_count = cc_count = mf_count = 0
    
    with open(input_ground_truth, 'r') as fin:
        # Open all output files
        with open(bp_file, 'w') as fbp, \
             open(cc_file, 'w') as fcc, \
             open(mf_file, 'w') as fmf:
            
            # Write headers
            header = "Protein_ID\tGround_Truth_GO_Terms\n"
            fbp.write(header)
            fcc.write(header)
            fmf.write(header)
            
            # Skip input header
            next(fin)
            
            for line in fin:
                parts = line.strip().split('\t')
                if len(parts) < 2 or parts[1] == 'NOT_FOUND':
                    continue
                    
                protein_id = parts[0]
                go_terms = parts[1].split(';')
                
                # Separate GO terms by ontology
                bp_terms = []
                cc_terms = []
                mf_terms = []
                
                for go_term in go_terms:
                    go_term = go_term.strip()
                    if not go_term:
                        continue
                        
                    # Use the GO tree to determine ontology
                    try:
                        # Check which root ontology this term belongs to
                        if go_tree.IsChildOf(go_term, "GO:0008150"):  # Biological Process
                            bp_terms.append(go_term)
                        elif go_tree.IsChildOf(go_term, "GO:0005575"):  # Cellular Component
                            cc_terms.append(go_term)
                        elif go_tree.IsChildOf(go_term, "GO:0003674"):  # Molecular Function
                            mf_terms.append(go_term)
                        # If it's one of the root terms themselves
                        elif go_term == "GO:0008150":
                            bp_terms.append(go_term)
                        elif go_term == "GO:0005575":
                            cc_terms.append(go_term)
                        elif go_term == "GO:0003674":
                            mf_terms.append(go_term)
                    except:
                        # If GO term lookup fails, skip it
                        print(f"Warning: Could not classify GO term {go_term}")
                        continue
                
                # Write to appropriate files if terms exist
                if bp_terms:
                    fbp.write(f"{protein_id}\t{';'.join(bp_terms)}\n")
                    bp_count += 1
                    
                if cc_terms:
                    fcc.write(f"{protein_id}\t{';'.join(cc_terms)}\n")
                    cc_count += 1
                    
                if mf_terms:
                    fmf.write(f"{protein_id}\t{';'.join(mf_terms)}\n")
                    mf_count += 1
    
    print(f"\nOntology filtering complete!")
    print(f"Biological Process (BP): {bp_count} proteins -> {bp_file}")
    print(f"Cellular Component (CC): {cc_count} proteins -> {cc_file}")
    print(f"Molecular Function (MF): {mf_count} proteins -> {mf_file}")

def main():
    if len(sys.argv) < 4:
        print("Usage: python filter_ontologies.py <obo_file> <ground_truth_file> <output_dir>")
        print("Example: python filter_ontologies.py GO_June_1_2025.obo consolidated_ground_truth.tsv filtered_ground_truth/")
        sys.exit(1)
    
    obo_file = sys.argv[1]
    ground_truth_file = sys.argv[2] 
    output_dir = sys.argv[3]
    
    filter_ground_truth_by_ontology(obo_file, ground_truth_file, output_dir)

if __name__ == "__main__":
    main()