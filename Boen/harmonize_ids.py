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
    """First pass: extract all UniProt IDs from AlphaFold filenames that we need to map"""
    input_pdb_dir = "/data/summer2020/Boen/benchmark_testing_pdbs"
    uniprot_ids = set()
    filename_to_uniprot = {}
    
    print(f"Scanning PDB files in {input_pdb_dir}")
    
    for filename in os.listdir(input_pdb_dir):
        if filename.endswith('.pdb') or filename.endswith('.pdb.gz'):
            # Extract the middle section (UniProt ID) from AlphaFold filename
            # Example: AF-B1ZEFJ7-F1-model_v4.pdb -> B1ZEFJ7
            base_name = filename.replace('.pdb.gz', '').replace('.pdb', '')
            
            # Split by '-' and get the middle part (UniProt ID)
            parts = base_name.split('-')
            if len(parts) >= 3 and parts[0] == 'AF':
                uniprot_id = parts[1]  # Extract the UniProt ID (e.g., B1ZEFJ7)
                uniprot_ids.add(uniprot_id)
                filename_to_uniprot[filename] = uniprot_id
                print(f"Extracted: {filename} -> {uniprot_id}")
            else:
                print(f"Warning: Unexpected filename format: {filename}")
    
    print(f"Found {len(uniprot_ids)} unique UniProt IDs to map")
    return uniprot_ids, filename_to_uniprot

def find_mappings_efficiently(target_uniprot_ids):
    """Efficiently search through the large mapping file for only the UniProt IDs we need"""
    mapping_file = "/data/shared/databases/UniProt2025/idmapping_uni.txt"
    found_mappings = {}
    
    print(f"Searching for mappings in {mapping_file} (this may take a while...)")
    print(f"Looking for these UniProt IDs: {sorted(list(target_uniprot_ids))[:10]}{'...' if len(target_uniprot_ids) > 10 else ''}")
    
    # Convert to set for O(1) lookup
    target_ids_set = set(target_uniprot_ids)
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
                    uniprot_id = parts[0].strip()  # First column is UniProt ID
                    mapped_value = parts[1].strip()  # Second column is the mapped value
                    
                    # Only store mapping if it's one we need
                    if uniprot_id in target_ids_set:
                        found_mappings[uniprot_id] = mapped_value
                        print(f"Found mapping: {uniprot_id} -> {mapped_value}")
                        
                        # Early exit if we found all mappings
                        if len(found_mappings) == len(target_ids_set):
                            print(f"Found all {len(found_mappings)} mappings after {lines_processed:,} lines")
                            break
    
    print(f"Finished processing {lines_processed:,} lines")
    print(f"Found {len(found_mappings)} out of {len(target_uniprot_ids)} requested mappings")
    
    return found_mappings

def rename_pdb_files_with_mappings(uniprot_mappings, filename_to_uniprot):
    """Rename PDB files using the found mappings"""
    input_pdb_dir = "/data/summer2020/Boen/benchmark_testing_pdbs"
    output_pdb_dir = "/data/summer2020/Boen/TransFun/data/benchmark_testing_pdbs_renamed"
    
    # Create output directory
    os.makedirs(output_pdb_dir, exist_ok=True)
    
    renamed_count = 0
    not_found_count = 0
    
    print(f"\nRenaming PDB files from {input_pdb_dir}")
    
    for filename in os.listdir(input_pdb_dir):
        if filename.endswith('.pdb') or filename.endswith('.pdb.gz'):
            # Get the UniProt ID we extracted for this filename
            if filename in filename_to_uniprot:
                original_uniprot_id = filename_to_uniprot[filename]
                
                # Check if we have a mapping for this UniProt ID
                if original_uniprot_id in uniprot_mappings:
                    mapped_value = uniprot_mappings[original_uniprot_id]
                    
                    # Determine file extension
                    extension = '.pdb.gz' if filename.endswith('.pdb.gz') else '.pdb'
                    new_filename = f"{mapped_value}{extension}"
                    
                    # Copy file with new name
                    src_path = os.path.join(input_pdb_dir, filename)
                    dst_path = os.path.join(output_pdb_dir, new_filename)
                    shutil.copy2(src_path, dst_path)
                    print(f"Renamed: {filename} -> {new_filename} ({original_uniprot_id} -> {mapped_value})")
                    renamed_count += 1
                else:
                    # Copy file as-is if no mapping found
                    src_path = os.path.join(input_pdb_dir, filename)
                    dst_path = os.path.join(output_pdb_dir, filename)
                    shutil.copy2(src_path, dst_path)
                    print(f"No mapping found for {original_uniprot_id}, keeping: {filename}")
                    not_found_count += 1
            else:
                # Handle unexpected filename format
                src_path = os.path.join(input_pdb_dir, filename)
                dst_path = os.path.join(output_pdb_dir, filename)
                shutil.copy2(src_path, dst_path)
                print(f"Unexpected filename format, keeping: {filename}")
                not_found_count += 1
    
    print(f"\nRenaming Summary:")
    print(f"Successfully renamed: {renamed_count} files")
    print(f"No mapping found: {not_found_count} files")
    print(f"Output directory: {output_pdb_dir}")

def create_mapping_report(uniprot_mappings, target_uniprot_ids, filename_to_uniprot):
    """Create a report of the mapping for verification"""
    report_file = "/data/summer2020/Boen/mapping_report.txt"
    
    with open(report_file, 'w') as f:
        f.write("PDB File Renaming Report\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Total UniProt IDs to map: {len(target_uniprot_ids)}\n")
        f.write(f"Successful mappings found: {len(uniprot_mappings)}\n")
        f.write(f"Missing mappings: {len(target_uniprot_ids) - len(uniprot_mappings)}\n\n")
        
        f.write("SUCCESSFUL MAPPINGS:\n")
        f.write("-" * 30 + "\n")
        for uniprot_id in sorted(uniprot_mappings.keys()):
            mapped_value = uniprot_mappings[uniprot_id]
            f.write(f"{uniprot_id} -> {mapped_value}\n")
        
        f.write("\nMISSING MAPPINGS:\n")
        f.write("-" * 30 + "\n")
        missing_ids = target_uniprot_ids - set(uniprot_mappings.keys())
        for uniprot_id in sorted(missing_ids):
            f.write(f"{uniprot_id} -> NO MAPPING FOUND\n")
            
        f.write("\nFILE RENAMING DETAILS:\n")
        f.write("-" * 30 + "\n")
        for filename, uniprot_id in sorted(filename_to_uniprot.items()):
            if uniprot_id in uniprot_mappings:
                mapped_value = uniprot_mappings[uniprot_id]
                extension = '.pdb.gz' if filename.endswith('.pdb.gz') else '.pdb'
                new_filename = f"{mapped_value}{extension}"
                f.write(f"{filename} -> {new_filename}\n")
            else:
                f.write(f"{filename} -> NO MAPPING (keeps original name)\n")
    
    print(f"Detailed mapping report saved to: {report_file}")

def main():
    print("UniProt ID Harmonization for TransFun (Large File Optimized)")
    print("=" * 60)
    
    # Step 1: Extract all UniProt IDs from PDB filenames that we need to map
    target_uniprot_ids, filename_to_uniprot = extract_alphafold_ids_from_pdbs()
    
    # Step 2: Efficiently search the large mapping file
    uniprot_mappings = find_mappings_efficiently(target_uniprot_ids)
    
    # Step 3: Create mapping report
    create_mapping_report(uniprot_mappings, target_uniprot_ids, filename_to_uniprot)
    
    # Step 4: Rename the files
    rename_pdb_files_with_mappings(uniprot_mappings, filename_to_uniprot)
    
    print("\nDone! Files are ready for TransFun prediction.")
    print("Check the mapping report for details on what was renamed.")

if __name__ == "__main__":
    main()