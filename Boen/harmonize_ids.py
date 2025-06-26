#!/usr/bin/env python3
"""
Script to rename PDB files in benchmark_testing_pdbs using UniProt ID mapping
Optimized for very large mapping files (250M+ lines)
Hardcoded for specific file locations
"""

import os
import shutil
from collections import defaultdict

def extract_alphafold_ids_from_pdbs():
    """First pass: extract all AlphaFold IDs we need to map"""
    input_pdb_dir = "/data/summer2020/Boen/benchmark_testing_pdbs"
    alphafold_ids = set()
    
    print(f"Scanning PDB files in {input_pdb_dir}")
    
    for filename in os.listdir(input_pdb_dir):
        if filename.endswith('.pdb') or filename.endswith('.pdb.gz'):
            # Extract AlphaFold ID from filename
            if filename.endswith('.pdb.gz'):
                alphafold_id = filename.replace('.pdb.gz', '')
            else:
                alphafold_id = filename.replace('.pdb', '')
            
            # Remove any additional suffixes like -F1-model_v4
            # Keep only the main AlphaFold ID (e.g., AF-Q6GZX4-F1)
            if '-F1-model' in alphafold_id:
                alphafold_id = alphafold_id.split('-F1-model')[0] + '-F1'
            
            alphafold_ids.add(alphafold_id)
    
    print(f"Found {len(alphafold_ids)} unique AlphaFold IDs to map")
    return alphafold_ids

def find_mappings_efficiently(target_ids):
    """Efficiently search through the large mapping file for only the IDs we need"""
    mapping_file = "/data/shared/databases/UniProt2025/idmapping_uni.txt"
    found_mappings = {}
    
    print(f"Searching for mappings in {mapping_file} (this may take a while...)")
    
    # Convert to set for O(1) lookup
    target_ids_set = set(target_ids)
    lines_processed = 0
    
    with open(mapping_file, 'r') as f:
        for line in f:
            lines_processed += 1
            
            # Progress indicator every 10M lines
            if lines_processed % 10_000_000 == 0:
                print(f"Processed {lines_processed:,} lines, found {len(found_mappings)} mappings so far")
            
            line = line.strip()
            if line:
                parts = line.split('\t')
                if len(parts) >= 2:
                    alphafold_id = parts[0].strip()
                    uniprot_id = parts[1].strip()
                    
                    # Only store mapping if it's one we need
                    if alphafold_id in target_ids_set:
                        found_mappings[alphafold_id] = uniprot_id
                        
                        # Early exit if we found all mappings
                        if len(found_mappings) == len(target_ids_set):
                            print(f"Found all {len(found_mappings)} mappings after {lines_processed:,} lines")
                            break
    
    print(f"Finished processing {lines_processed:,} lines")
    print(f"Found {len(found_mappings)} out of {len(target_ids)} requested mappings")
    
    return found_mappings

def rename_pdb_files_with_mappings(id_mapping):
    """Rename PDB files using the found mappings"""
    input_pdb_dir = "/data/summer2020/Boen/benchmark_testing_pdbs"
    output_pdb_dir = "/data/summer2020/Boen/benchmark_testing_pdbs_renamed"
    
    # Create output directory
    os.makedirs(output_pdb_dir, exist_ok=True)
    
    renamed_count = 0
    not_found_count = 0
    
    print(f"\nRenaming PDB files from {input_pdb_dir}")
    
    for filename in os.listdir(input_pdb_dir):
        if filename.endswith('.pdb') or filename.endswith('.pdb.gz'):
            # Extract AlphaFold ID from filename
            if filename.endswith('.pdb.gz'):
                alphafold_id = filename.replace('.pdb.gz', '')
                extension = '.pdb.gz'
            else:
                alphafold_id = filename.replace('.pdb', '')
                extension = '.pdb'
            
            # Clean up AlphaFold ID
            if '-F1-model' in alphafold_id:
                alphafold_id = alphafold_id.split('-F1-model')[0] + '-F1'
            
            # Check if we have a mapping for this ID
            if alphafold_id in id_mapping:
                uniprot_id = id_mapping[alphafold_id]
                new_filename = f"{uniprot_id}{extension}"
                
                # Copy file with new name
                src_path = os.path.join(input_pdb_dir, filename)
                dst_path = os.path.join(output_pdb_dir, new_filename)
                shutil.copy2(src_path, dst_path)
                print(f"Renamed: {filename} -> {new_filename}")
                renamed_count += 1
            else:
                # Copy file as-is if no mapping found
                src_path = os.path.join(input_pdb_dir, filename)
                dst_path = os.path.join(output_pdb_dir, filename)
                shutil.copy2(src_path, dst_path)
                print(f"No mapping found for {alphafold_id}, keeping: {filename}")
                not_found_count += 1
    
    print(f"\nRenaming Summary:")
    print(f"Successfully renamed: {renamed_count} files")
    print(f"No mapping found: {not_found_count} files")
    print(f"Output directory: {output_pdb_dir}")

def create_mapping_report(id_mapping, target_ids):
    """Create a report of the mapping for verification"""
    report_file = "/data/summer2020/Boen/mapping_report.txt"
    
    with open(report_file, 'w') as f:
        f.write("PDB File Renaming Report\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Total AlphaFold IDs to map: {len(target_ids)}\n")
        f.write(f"Successful mappings found: {len(id_mapping)}\n")
        f.write(f"Missing mappings: {len(target_ids) - len(id_mapping)}\n\n")
        
        f.write("SUCCESSFUL MAPPINGS:\n")
        f.write("-" * 30 + "\n")
        for alphafold_id in sorted(id_mapping.keys()):
            uniprot_id = id_mapping[alphafold_id]
            f.write(f"{alphafold_id} -> {uniprot_id}\n")
        
        f.write("\nMISSING MAPPINGS:\n")
        f.write("-" * 30 + "\n")
        missing_ids = target_ids - set(id_mapping.keys())
        for alphafold_id in sorted(missing_ids):
            f.write(f"{alphafold_id} -> NO MAPPING FOUND\n")
    
    print(f"Detailed mapping report saved to: {report_file}")

def main():
    print("UniProt ID Harmonization for TransFun (Large File Optimized)")
    print("=" * 60)
    
    # Step 1: Extract all AlphaFold IDs we need to map
    target_ids = extract_alphafold_ids_from_pdbs()
    
    # Step 2: Efficiently search the large mapping file
    id_mapping = find_mappings_efficiently(target_ids)
    
    # Step 3: Create mapping report
    create_mapping_report(id_mapping, target_ids)
    
    # Step 4: Rename the files
    rename_pdb_files_with_mappings(id_mapping)
    
    print("\nDone! Files are ready for TransFun prediction.")
    print("Check the mapping report for details on what was renamed.")

if __name__ == "__main__":
    main()