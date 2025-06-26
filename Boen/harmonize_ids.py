#!/usr/bin/env python3
"""
Script to rename PDB files in benchmark_testing_pdbs using UniProt ID mapping
Hardcoded for specific file locations
"""

import os
import shutil

def load_uniprot_mapping():
    """Load UniProt ID mapping from hardcoded location"""
    mapping_file = "/data/shared/databases/UniProt2025/idmapping_uni.txt"
    id_mapping = {}
    
    print(f"Loading ID mapping from {mapping_file}")
    
    with open(mapping_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                parts = line.split('\t')
                if len(parts) >= 2:
                    # Format: AlphaFold_ID -> UniProt_ID
                    alphafold_id = parts[0].strip()
                    uniprot_id = parts[1].strip()
                    id_mapping[alphafold_id] = uniprot_id
    
    print(f"Loaded {len(id_mapping)} ID mappings")
    return id_mapping

def rename_pdb_files():
    """Rename PDB files in benchmark_testing_pdbs directory"""
    input_pdb_dir = "/data/summer2020/Boen/benchmark_testing_pdbs"
    output_pdb_dir = "/data/summer2020/Boen/benchmark_testing_pdbs_renamed"
    
    # Load the mapping
    id_mapping = load_uniprot_mapping()
    
    # Create output directory
    os.makedirs(output_pdb_dir, exist_ok=True)
    
    renamed_count = 0
    not_found_count = 0
    
    print(f"\nProcessing PDB files from {input_pdb_dir}")
    
    for filename in os.listdir(input_pdb_dir):
        if filename.endswith('.pdb') or filename.endswith('.pdb.gz'):
            # Extract AlphaFold ID from filename
            if filename.endswith('.pdb.gz'):
                alphafold_id = filename.replace('.pdb.gz', '')
                extension = '.pdb.gz'
            else:
                alphafold_id = filename.replace('.pdb', '')
                extension = '.pdb'
            
            # Remove any additional suffixes like -F1-model_v4
            # Keep only the main AlphaFold ID (e.g., AF-Q6GZX4-F1)
            if '-F1-model' in alphafold_id:
                alphafold_id = alphafold_id.split('-F1-model')[0] + '-F1'
            
            # Check if this ID needs mapping
            if alphafold_id in id_mapping:
                uniprot_id = id_mapping[alphafold_id]
                new_filename = f"{uniprot_id}{extension}"
                
                # Copy file with new name
                src_path = os.path.join(input_pdb_dir, filename)
                dst_path = os.path.join(output_pdb_dir, new_filename)
                shutil.copy2(src_path, dst_path)
                print(f"Renamed: {filename} -> {new_filename} (AlphaFold: {alphafold_id} -> UniProt: {uniprot_id})")
                renamed_count += 1
            else:
                # Copy file as-is if no mapping found
                src_path = os.path.join(input_pdb_dir, filename)
                dst_path = os.path.join(output_pdb_dir, filename)
                shutil.copy2(src_path, dst_path)
                print(f"No mapping found for {alphafold_id}, keeping original name: {filename}")
                not_found_count += 1
    
    print(f"\nSummary:")
    print(f"Successfully renamed: {renamed_count} files")
    print(f"No mapping found: {not_found_count} files")
    print(f"Output directory: {output_pdb_dir}")

def create_mapping_report():
    """Create a report of the mapping for verification"""
    input_pdb_dir = "/data/summer2020/Boen/benchmark_testing_pdbs"
    report_file = "/data/summer2020/Boen/mapping_report.txt"
    
    id_mapping = load_uniprot_mapping()
    
    with open(report_file, 'w') as f:
        f.write("PDB File Renaming Report\n")
        f.write("=" * 50 + "\n\n")
        
        for filename in sorted(os.listdir(input_pdb_dir)):
            if filename.endswith('.pdb') or filename.endswith('.pdb.gz'):
                if filename.endswith('.pdb.gz'):
                    alphafold_id = filename.replace('.pdb.gz', '')
                    extension = '.pdb.gz'
                else:
                    alphafold_id = filename.replace('.pdb', '')
                    extension = '.pdb'
                
                # Clean up AlphaFold ID
                if '-F1-model' in alphafold_id:
                    alphafold_id = alphafold_id.split('-F1-model')[0] + '-F1'
                
                if alphafold_id in id_mapping:
                    uniprot_id = id_mapping[alphafold_id]
                    new_filename = f"{uniprot_id}{extension}"
                    f.write(f"{filename} -> {new_filename}\n")
                    f.write(f"  AlphaFold ID: {alphafold_id}\n")
                    f.write(f"  UniProt ID: {uniprot_id}\n\n")
                else:
                    f.write(f"{filename} -> NO MAPPING FOUND\n")
                    f.write(f"  AlphaFold ID: {alphafold_id}\n\n")
    
    print(f"Mapping report saved to: {report_file}")

def main():
    print("UniProt ID Harmonization for TransFun")
    print("=" * 40)
    
    # Create mapping report first
    create_mapping_report()
    
    # Rename the files
    rename_pdb_files()
    
    print("\nDone! Files are ready for TransFun prediction.")
    print("Check the mapping report for details on what was renamed.")

if __name__ == "__main__":
    main()