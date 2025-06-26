import os
import subprocess
from Bio import SeqIO
import click as ck
import shutil

# ==============================================================================
# --- CONFIGURATION ---
# ==============================================================================

# 1. Path to the InterProScan executable shell script.
INTERPROSCAN_PATH = "/data/shared/tools/interproscan-5.75-106.0/interproscan.sh"

# 2. Path to the specific failed chunk file you want to analyze.
FAILED_CHUNK_FILE = "/data/summer2020/Boen/deepgozero_predictions/fasta_chunks/chunk_131.fasta"

# 3. Directory to store temporary files for this script.
TEMP_DIAGNOSTIC_DIR = "/data/summer2020/Boen/deepgozero_predictions/diagnostic_run"

# ==============================================================================
# --- SCRIPT LOGIC ---
# ==============================================================================

@ck.command()
def main():
    """
    Analyzes a single failed FASTA chunk by running InterProScan on each
    protein individually to isolate the exact sequence causing the crash.
    """
    if not os.path.exists(FAILED_CHUNK_FILE):
        print(f"ERROR: Failed chunk file not found at {FAILED_CHUNK_FILE}")
        return

    os.makedirs(TEMP_DIAGNOSTIC_DIR, exist_ok=True)
    
    chunk_name = os.path.basename(FAILED_CHUNK_FILE)
    print(f"--- Starting Isolation Test for: {chunk_name} ---\n")
    
    records = list(SeqIO.parse(FAILED_CHUNK_FILE, "fasta"))

    for i, record in enumerate(records):
        protein_id = record.id
        single_protein_fasta = os.path.join(TEMP_DIAGNOSTIC_DIR, f"{protein_id.replace('|','_')}.fasta")
        
        # Write the single protein to its own file
        SeqIO.write(record, single_protein_fasta, "fasta")
        
        print(f"Testing protein {i+1}/{len(records)}: {protein_id} ...")

        # Clean up any old InterProScan temp directories before each run
        if os.path.exists("temp"):
            shutil.rmtree("temp")

        # Define the InterProScan command
        cmd = [
            INTERPROSCAN_PATH,
            "-i", single_protein_fasta,
            "-f", "TSV",
            "--goterms"
            # We run the full analysis to replicate the failure condition
        ]

        try:
            # Run InterProScan on the single protein
            subprocess.run(
                cmd, 
                check=True,        # This will raise an error if the command fails
                capture_output=True, # This hides the verbose output unless there's an error
                text=True
            )
            print(f"  -> SUCCESS: {protein_id} processed cleanly.\n")

        except subprocess.CalledProcessError as e:
            # If we get here, this is the protein that caused the crash
            print("\n" + "="*80)
            print(f"FOUND THE PROBLEMATIC PROTEIN: {protein_id}")
            print("="*80)
            print("\nThis protein caused InterProScan to fail. Please remove it from your main")
            print(f"FASTA file ('{os.path.basename(FAILED_CHUNK_FILE)}' came from it) and restart the main pipeline.")
            print("\nFull error message from InterProScan:")
            print(e.stderr)
            # Stop the script once the problematic protein is found
            return
        
        finally:
            # Clean up the single-protein file
            if os.path.exists(single_protein_fasta):
                os.remove(single_protein_fasta)

    print("--- Analysis Complete ---")
    print("All proteins in the chunk were processed successfully individually.")
    print("This indicates the failure might be due to a combination of proteins or a memory issue when running them together.")
    
    # Clean up the diagnostic directory
    if os.path.exists(TEMP_DIAGNOSTIC_DIR):
        shutil.rmtree(TEMP_DIAGNOSTIC_DIR)


if __name__ == "__main__":
    main()
