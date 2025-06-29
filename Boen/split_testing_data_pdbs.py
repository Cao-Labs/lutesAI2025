import os
import shutil
from Bio import SeqIO
from collections import defaultdict

# --- Configuration ---
# Source directory containing the many individual .pdb files
SRC_PDB_DIR = "/data/summer2020/naufal/testing_pdbs"

# Destination directory for the new benchmark set of PDBs
DEST_PDB_DIR = "/data/summer2020/Boen/benchmark_testing_pdbs"

# Final renamed directory for TransFun
RENAMED_PDB_DIR = "/data/summer2020/Boen/TransFun/data/benchmark_testing_pdbs_renamed"

# FASTA file containing the sequences you want to predict
FASTA_FILE = "/data/summer2020/Boen/TransFun/data/benchmark_testing_sequences.fasta"

# UniProt ID mapping file
MAPPING_FILE = "/data/shared/databases/UniProt2025/idmapping_uni.txt"

def extract_fasta_protein_ids():
    """Extract protein IDs from FASTA file"""
    try:
        fasta_proteins = set()
        print(f"Reading protein IDs from FASTA file: {FASTA_FILE}")
        
        for record in SeqIO.parse(FASTA_FILE, "fasta"):
            # Extract protein ID (everything before first space or pipe)
            protein_id = record.id.split('|')[0].split()[0]
            fasta_proteins.add(protein_id)
        
        print(f"Found {len(fasta_proteins)} unique protein IDs in FASTA file")
        print(f"First 10 protein IDs: {list(fasta_proteins)[:10]}")
        
        return fasta_proteins
        
    except FileNotFoundError:
        print(f"Error: FASTA file not found at {FASTA_FILE}")
        return set()
    except Exception as e:
        print(f"Error reading FASTA file: {e}")
        return set()

def extract_uniprot_ids_from_pdbs():
    """Extract UniProt IDs from AlphaFold PDB filenames in source directory"""
    uniprot_ids = set()
    filename_to_uniprot = {}
    
    print(f"Scanning PDB files in {SRC_PDB_DIR}")
    
    try:
        for filename in os.listdir(SRC_PDB_DIR):
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
                else:
                    # For non-AlphaFold files, use the base filename as ID
                    protein_id = base_name
                    uniprot_ids.add(protein_id)
                    filename_to_uniprot[filename] = protein_id
        
        print(f"Found {len(uniprot_ids)} unique protein IDs from PDB files")
        print(f"First 10 PDB protein IDs: {list(uniprot_ids)[:10]}")
        
        return uniprot_ids, filename_to_uniprot
        
    except FileNotFoundError:
        print(f"Error: Source PDB directory not found at {SRC_PDB_DIR}")
        return set(), {}

def find_uniprot_mappings(target_uniprot_ids, fasta_protein_ids):
    """
    Find mappings from UniProt IDs to FASTA protein IDs using the mapping file
    This creates a mapping: uniprot_id -> fasta_protein_id
    """
    found_mappings = {}
    reverse_mappings = {}  # fasta_id -> uniprot_id
    
    print(f"Searching for mappings in {MAPPING_FILE}")
    print(f"Looking for mappings from {len(target_uniprot_ids)} UniProt IDs to {len(fasta_protein_ids)} FASTA IDs")
    
    # Convert to sets for O(1) lookup
    target_ids_set = set(target_uniprot_ids)
    fasta_ids_set = set(fasta_protein_ids)
    lines_processed = 0
    
    try:
        with open(MAPPING_FILE, 'r') as f:
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
                        
                        # Check if this UniProt ID is in our PDB files AND the mapped value is in our FASTA
                        if uniprot_id in target_ids_set and mapped_value in fasta_ids_set:
                            found_mappings[uniprot_id] = mapped_value
                            reverse_mappings[mapped_value] = uniprot_id
                            print(f"Found useful mapping: {uniprot_id} -> {mapped_value}")
                            
                            # Early exit if we found mappings for all FASTA proteins
                            if len(reverse_mappings) == len(fasta_ids_set):
                                print(f"Found mappings for all FASTA proteins after {lines_processed:,} lines")
                                break
        
        print(f"Finished processing {lines_processed:,} lines")
        print(f"Found {len(found_mappings)} useful mappings")
        
        return found_mappings, reverse_mappings
        
    except FileNotFoundError:
        print(f"Error: Mapping file not found at {MAPPING_FILE}")
        return {}, {}

def select_and_copy_matching_pdbs(fasta_protein_ids, pdb_uniprot_ids, filename_to_uniprot, uniprot_mappings):
    """
    Select and copy PDB files that can be mapped to FASTA protein IDs
    """
    print(f"Selecting PDB files that match FASTA proteins...")
    
    # Create destination directory
    os.makedirs(DEST_PDB_DIR, exist_ok=True)
    
    # Find which PDB files we need
    needed_pdbs = []
    direct_matches = 0
    mapped_matches = 0
    
    for fasta_id in fasta_protein_ids:
        # Check for direct matches (PDB protein ID = FASTA protein ID)
        for filename, pdb_protein_id in filename_to_uniprot.items():
            if pdb_protein_id == fasta_id:
                needed_pdbs.append((filename, fasta_id, "direct"))
                direct_matches += 1
                break
        else:
            # Check for mapped matches (UniProt ID maps to FASTA ID)
            for uniprot_id, mapped_fasta_id in uniprot_mappings.items():
                if mapped_fasta_id == fasta_id:
                    # Find the PDB file with this UniProt ID
                    for filename, pdb_uniprot_id in filename_to_uniprot.items():
                        if pdb_uniprot_id == uniprot_id:
                            needed_pdbs.append((filename, fasta_id, "mapped"))
                            mapped_matches += 1
                            break
                    break
    
    print(f"Match analysis:")
    print(f"  - Direct matches: {direct_matches}")
    print(f"  - Mapped matches: {mapped_matches}")
    print(f"  - Total matches: {len(needed_pdbs)}")
    print(f"  - FASTA proteins without PDB: {len(fasta_protein_ids) - len(needed_pdbs)}")
    
    # Copy the needed PDB files
    copied_count = 0
    for filename, fasta_id, match_type in needed_pdbs:
        src_path = os.path.join(SRC_PDB_DIR, filename)
        dst_path = os.path.join(DEST_PDB_DIR, filename)
        
        try:
            shutil.copy2(src_path, dst_path)
            copied_count += 1
            print(f"Copied ({match_type}): {filename} -> {fasta_id}")
        except Exception as e:
            print(f"Failed to copy {filename}: {e}")
    
    print(f"Successfully copied {copied_count} PDB files")
    return needed_pdbs

def rename_pdbs_for_transfun(needed_pdbs, filename_to_uniprot, uniprot_mappings):
    """
    Rename the copied PDB files to match FASTA protein IDs for TransFun
    """
    print(f"Renaming PDB files for TransFun...")
    
    # Create renamed directory
    os.makedirs(RENAMED_PDB_DIR, exist_ok=True)
    
    renamed_count = 0
    
    for filename, fasta_id, match_type in needed_pdbs:
        # Determine file extension
        extension = '.pdb.gz' if filename.endswith('.pdb.gz') else '.pdb'
        new_filename = f"{fasta_id}{extension}"
        
        # Copy and rename file
        src_path = os.path.join(DEST_PDB_DIR, filename)
        dst_path = os.path.join(RENAMED_PDB_DIR, new_filename)
        
        try:
            shutil.copy2(src_path, dst_path)
            print(f"Renamed: {filename} -> {new_filename} ({match_type})")
            renamed_count += 1
        except Exception as e:
            print(f"Failed to rename {filename}: {e}")
    
    print(f"Successfully renamed {renamed_count} files")

def create_comprehensive_report(fasta_protein_ids, pdb_uniprot_ids, uniprot_mappings, needed_pdbs):
    """Create a comprehensive report of the matching process"""
    report_file = "/data/summer2020/Boen/pdb_matching_report.txt"
    
    with open(report_file, 'w') as f:
        f.write("PDB-FASTA Matching Report\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("SUMMARY:\n")
        f.write(f"FASTA proteins: {len(fasta_protein_ids)}\n")
        f.write(f"Available PDB proteins: {len(pdb_uniprot_ids)}\n")
        f.write(f"UniProt mappings found: {len(uniprot_mappings)}\n")
        f.write(f"Successful matches: {len(needed_pdbs)}\n")
        f.write(f"Missing proteins: {len(fasta_protein_ids) - len(needed_pdbs)}\n\n")
        
        f.write("SUCCESSFUL MATCHES:\n")
        f.write("-" * 30 + "\n")
        for filename, fasta_id, match_type in sorted(needed_pdbs):
            extension = '.pdb.gz' if filename.endswith('.pdb.gz') else '.pdb'
            new_filename = f"{fasta_id}{extension}"
            f.write(f"{filename} -> {new_filename} ({match_type} match)\n")
        
        f.write("\nUNIPROT MAPPINGS USED:\n")
        f.write("-" * 30 + "\n")
        for uniprot_id, mapped_value in sorted(uniprot_mappings.items()):
            f.write(f"{uniprot_id} -> {mapped_value}\n")
        
        f.write("\nMISSING FASTA PROTEINS:\n")
        f.write("-" * 30 + "\n")
        matched_fasta_ids = {fasta_id for _, fasta_id, _ in needed_pdbs}
        missing_fasta_ids = fasta_protein_ids - matched_fasta_ids
        for fasta_id in sorted(missing_fasta_ids):
            f.write(f"{fasta_id} -> NO PDB FOUND\n")
    
    print(f"Comprehensive report saved to: {report_file}")

def main():
    print("PDB-FASTA Matching with UniProt ID Mapping")
    print("=" * 50)
    
    # Step 1: Extract protein IDs from FASTA file
    fasta_protein_ids = extract_fasta_protein_ids()
    if not fasta_protein_ids:
        print("No FASTA proteins found. Exiting.")
        return
    
    # Step 2: Extract UniProt IDs from PDB files
    pdb_uniprot_ids, filename_to_uniprot = extract_uniprot_ids_from_pdbs()
    if not pdb_uniprot_ids:
        print("No PDB files found. Exiting.")
        return
    
    # Step 3: Find mappings between UniProt IDs and FASTA protein IDs
    uniprot_mappings, reverse_mappings = find_uniprot_mappings(pdb_uniprot_ids, fasta_protein_ids)
    
    # Step 4: Select and copy matching PDB files
    needed_pdbs = select_and_copy_matching_pdbs(fasta_protein_ids, pdb_uniprot_ids, filename_to_uniprot, uniprot_mappings)
    
    if not needed_pdbs:
        print("No matching PDB files found. Check if protein naming conventions match.")
        return
    
    # Step 5: Rename PDB files for TransFun
    rename_pdbs_for_transfun(needed_pdbs, filename_to_uniprot, uniprot_mappings)
    
    # Step 6: Create comprehensive report
    create_comprehensive_report(fasta_protein_ids, pdb_uniprot_ids, uniprot_mappings, needed_pdbs)
    
    print(f"\nSUCCESS! Process completed.")
    print(f"Original PDBs: {DEST_PDB_DIR}")
    print(f"Renamed PDBs for TransFun: {RENAMED_PDB_DIR}")
    print(f"These renamed PDB files should now match your FASTA protein IDs.")

if __name__ == "__main__":
    main()