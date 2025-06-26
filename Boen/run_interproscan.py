import os
import subprocess
import pandas as pd
from datetime import datetime
from Bio import SeqIO
import click as ck
import time
import multiprocessing
import tempfile
import shutil

# ==============================================================================
# --- CONFIGURATION ---
# YOU MUST EDIT THESE PATHS TO MATCH YOUR SYSTEM
# ==============================================================================

# 1. Path to your input protein sequences
FASTA_FILE = "/data/summer2020/Boen/benchmark_testing_sequences.fasta"

# 2. Path to the main InterProScan executable shell script
INTERPROSCAN_PATH = "/data/shared/tools/interproscan-5.75-106.0/interproscan.sh"

# 3. A directory to store all intermediate and final output files
OUTPUT_DIR = "/data/summer2020/Boen/deepgozero_pipeline_output"

# 4. Number of CPU cores for InterProScan
CPU_CORES = "8"

# 5. Batch size for processing sequences (reduce if memory issues)
BATCH_SIZE = 100

# ==============================================================================
# --- SCRIPT LOGIC ---
# ==============================================================================

STATUS_FILE = os.path.join(OUTPUT_DIR, "interproscan_status.txt")
PROBLEMATIC_PROTEINS_FILE = os.path.join(OUTPUT_DIR, "problematic_proteins.txt")
SUCCESSFUL_PROTEINS_FILE = os.path.join(OUTPUT_DIR, "successful_proteins.txt")

def update_status(message):
    """Writes a timestamped message to the status file."""
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with open(STATUS_FILE, "a") as f:
        f.write(f"{now} - {message}\n")
    print(message)

def count_fasta_proteins(fasta_file):
    """Counts the number of sequences in a FASTA file."""
    try:
        return sum(1 for _ in SeqIO.parse(fasta_file, "fasta"))
    except FileNotFoundError:
        return 0

def validate_protein_sequence(record):
    """Validates a protein sequence for common issues."""
    issues = []
    sequence = str(record.seq).upper()
    
    # Check for minimum length
    if len(sequence) < 10:
        issues.append("Too short (< 10 amino acids)")
    
    # Check for maximum length (InterProScan might have issues with very long sequences)
    if len(sequence) > 50000:
        issues.append("Too long (> 50,000 amino acids)")
    
    # Check for invalid amino acid characters
    valid_aa = set("ACDEFGHIKLMNPQRSTVWY*X-")
    invalid_chars = set(sequence) - valid_aa
    if invalid_chars:
        issues.append(f"Invalid characters: {', '.join(sorted(invalid_chars))}")
    
    # Check for excessive ambiguous residues
    ambiguous_count = sequence.count('X')
    if ambiguous_count / len(sequence) > 0.5:
        issues.append(f"Too many ambiguous residues: {ambiguous_count}/{len(sequence)} (>{50}%)")
    
    # Check for sequences that are mostly gaps or stops
    gap_stop_count = sequence.count('-') + sequence.count('*')
    if gap_stop_count / len(sequence) > 0.3:
        issues.append(f"Too many gaps/stops: {gap_stop_count}/{len(sequence)} (>{30}%)")
    
    return issues

def create_batches(sequences, batch_size):
    """Creates batches of sequences for processing."""
    batches = []
    for i in range(0, len(sequences), batch_size):
        batches.append(sequences[i:i + batch_size])
    return batches

def run_interproscan_batch(batch_sequences, batch_num, total_batches):
    """Runs InterProScan on a batch of sequences."""
    update_status(f"Processing batch {batch_num}/{total_batches} ({len(batch_sequences)} sequences)")
    
    # Create temporary files for this batch
    temp_dir = tempfile.mkdtemp(prefix=f"interpro_batch_{batch_num}_")
    temp_fasta = os.path.join(temp_dir, f"batch_{batch_num}.fasta")
    temp_output = os.path.join(temp_dir, f"batch_{batch_num}_results.tsv")
    
    try:
        # Write batch sequences to temporary FASTA file
        with open(temp_fasta, 'w') as f:
            for record in batch_sequences:
                f.write(f">{record.id}\n{record.seq}\n")
        
        # Run InterProScan
        cmd = [
            INTERPROSCAN_PATH, "-i", temp_fasta, "-f", "TSV", "-o", temp_output,
            "--goterms", "-cpu", CPU_CORES
        ]
        
        process = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)  # 1 hour timeout
        
        if process.returncode == 0:
            # Success - read results
            results = []
            if os.path.exists(temp_output):
                with open(temp_output, 'r') as f:
                    results = f.readlines()
            
            successful_ids = set()
            for line in results:
                if line.strip():
                    successful_ids.add(line.split('\t')[0])
            
            # Identify failed sequences in this batch
            failed_sequences = []
            for record in batch_sequences:
                if record.id not in successful_ids:
                    failed_sequences.append(record)
            
            return results, failed_sequences, None
        else:
            # Batch failed - all sequences are problematic
            error_msg = f"Batch {batch_num} failed: {process.stderr}"
            return [], batch_sequences, error_msg
            
    except subprocess.TimeoutExpired:
        error_msg = f"Batch {batch_num} timed out after 1 hour"
        return [], batch_sequences, error_msg
    except Exception as e:
        error_msg = f"Batch {batch_num} error: {str(e)}"
        return [], batch_sequences, error_msg
    finally:
        # Clean up temporary files
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

def process_individual_sequences(failed_sequences):
    """Process failed sequences individually to identify specific problems."""
    update_status(f"Processing {len(failed_sequences)} failed sequences individually...")
    
    individual_results = []
    final_failed = []
    
    for i, record in enumerate(failed_sequences):
        update_status(f"Testing individual sequence {i+1}/{len(failed_sequences)}: {record.id}")
        
        temp_dir = tempfile.mkdtemp(prefix=f"interpro_individual_{record.id}_")
        temp_fasta = os.path.join(temp_dir, f"{record.id}.fasta")
        temp_output = os.path.join(temp_dir, f"{record.id}_results.tsv")
        
        try:
            # Write single sequence to file
            with open(temp_fasta, 'w') as f:
                f.write(f">{record.id}\n{record.seq}\n")
            
            # Run InterProScan with shorter timeout for individual sequences
            cmd = [
                INTERPROSCAN_PATH, "-i", temp_fasta, "-f", "TSV", "-o", temp_output,
                "--goterms", "-cpu", "1"  # Use single CPU for individual sequences
            ]
            
            process = subprocess.run(cmd, capture_output=True, text=True, timeout=900)  # 15 min timeout
            
            if process.returncode == 0 and os.path.exists(temp_output):
                with open(temp_output, 'r') as f:
                    results = f.readlines()
                individual_results.extend(results)
                update_status(f"✓ Successfully processed {record.id}")
            else:
                # This sequence is truly problematic
                error_reason = f"Exit code: {process.returncode}, STDERR: {process.stderr[:200]}"
                final_failed.append((record, error_reason))
                update_status(f"✗ Failed to process {record.id}: {error_reason}")
                
        except subprocess.TimeoutExpired:
            final_failed.append((record, "Individual processing timeout (15 min)"))
            update_status(f"✗ Timeout processing {record.id}")
        except Exception as e:
            final_failed.append((record, f"Exception: {str(e)}"))
            update_status(f"✗ Exception processing {record.id}: {str(e)}")
        finally:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
    
    return individual_results, final_failed

def analyze_problematic_proteins(failed_proteins_with_reasons, validation_issues):
    """Analyzes and categorizes problematic proteins."""
    update_status("Analyzing problematic proteins...")
    
    categories = {
        "sequence_validation_issues": [],
        "interpro_timeout": [],
        "interpro_memory_issues": [],
        "interpro_format_issues": [],
        "interpro_other_errors": []
    }
    
    # Categorize validation issues
    for protein_id, issues in validation_issues.items():
        if issues:
            categories["sequence_validation_issues"].append((protein_id, issues))
    
    # Categorize InterProScan failures
    for record, error_reason in failed_proteins_with_reasons:
        if "timeout" in error_reason.lower():
            categories["interpro_timeout"].append((record.id, error_reason))
        elif "memory" in error_reason.lower() or "outofmemory" in error_reason.lower():
            categories["interpro_memory_issues"].append((record.id, error_reason))
        elif "format" in error_reason.lower() or "parse" in error_reason.lower():
            categories["interpro_format_issues"].append((record.id, error_reason))
        else:
            categories["interpro_other_errors"].append((record.id, error_reason))
    
    return categories

@ck.command()
@ck.option('--force-rerun', is_flag=True, help="Force re-running InterProScan even if output files exist.")
@ck.option('--batch-size', default=BATCH_SIZE, help="Number of sequences to process in each batch.")
@ck.option('--ignore-validation-issues', is_flag=True, help="Automatically filter out proteins with validation issues (length, invalid characters, etc.)")
def main(force_rerun, batch_size, ignore_validation_issues):
    """Main function to run InterProScan with error handling."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Clear status file
    with open(STATUS_FILE, "w") as f:
        f.write("")
    
    update_status("InterProScan pipeline starting...")
    
    interpro_output_tsv = os.path.join(OUTPUT_DIR, "interproscan_results.tsv")
    
    if not force_rerun and os.path.exists(interpro_output_tsv):
        update_status("InterProScan results already exist. Use --force-rerun to regenerate.")
        return
    
    # Load and validate sequences
    update_status(f"Loading sequences from {FASTA_FILE}")
    sequences = list(SeqIO.parse(FASTA_FILE, "fasta"))
    total_proteins = len(sequences)
    
    if total_proteins == 0:
        update_status(f"ERROR: No proteins found in FASTA file: {FASTA_FILE}")
        return
    
    update_status(f"Found {total_proteins} proteins in input file.")
    
    # Validate sequences
    update_status("Validating protein sequences...")
    validation_issues = {}
    valid_sequences = []
    filtered_sequences = []
    
    for record in sequences:
        issues = validate_protein_sequence(record)
        if issues:
            validation_issues[record.id] = issues
            if ignore_validation_issues:
                filtered_sequences.append(record)
                if len(filtered_sequences) <= 10:  # Only show first 10 for brevity
                    update_status(f"FILTERED: {record.id} - {'; '.join(issues)}")
                elif len(filtered_sequences) == 11:
                    update_status("... (additional filtered sequences not shown)")
            else:
                update_status(f"Validation issues for {record.id}: {'; '.join(issues)}")
        else:
            valid_sequences.append(record)
    
    if ignore_validation_issues and filtered_sequences:
        update_status(f"Filtered out {len(filtered_sequences)} proteins with validation issues")
        update_status(f"Validation complete: {len(valid_sequences)} valid sequences will be processed")
        # Only track InterProScan-specific failures when ignoring validation issues
        validation_issues = {}  # Clear validation issues so they're not reported as problematic
    else:
        update_status(f"Validation complete: {len(valid_sequences)} valid, {len(validation_issues)} with issues")
    
    # Process valid sequences in batches
    if valid_sequences:
        batches = create_batches(valid_sequences, batch_size)
        update_status(f"Processing {len(valid_sequences)} sequences in {len(batches)} batches")
        
        all_results = []
        all_failed_sequences = []
        
        for i, batch in enumerate(batches, 1):
            results, failed_seqs, error_msg = run_interproscan_batch(batch, i, len(batches))
            all_results.extend(results)
            all_failed_sequences.extend(failed_seqs)
            
            if error_msg:
                update_status(f"Batch {i} error: {error_msg}")
        
        # Process individually failed sequences
        if all_failed_sequences:
            individual_results, final_failed = process_individual_sequences(all_failed_sequences)
            all_results.extend(individual_results)
        else:
            final_failed = []
        
        # Write successful results
        if all_results:
            with open(interpro_output_tsv, 'w') as f:
                f.writelines(all_results)
            update_status(f"Wrote {len(all_results)} InterProScan result lines to {interpro_output_tsv}")
        
        # Analyze and report problematic proteins
        problematic_analysis = analyze_problematic_proteins(final_failed, validation_issues)
        
        # Write detailed report
        with open(PROBLEMATIC_PROTEINS_FILE, 'w') as f:
            f.write("PROBLEMATIC PROTEINS ANALYSIS\n")
            f.write("=" * 50 + "\n\n")
            
            if ignore_validation_issues and filtered_sequences:
                f.write(f"FILTERED OUT (validation issues): {len(filtered_sequences)} proteins\n")
                f.write("These proteins were automatically excluded due to validation issues:\n")
                for record in filtered_sequences:
                    issues = validate_protein_sequence(record)
                    f.write(f"  {record.id}: {'; '.join(issues)}\n")
                f.write("\n")
            
            total_problematic = 0
            for category, proteins in problematic_analysis.items():
                if proteins:
                    f.write(f"{category.upper().replace('_', ' ')} ({len(proteins)} proteins):\n")
                    f.write("-" * 40 + "\n")
                    for protein_info in proteins:
                        if isinstance(protein_info[1], list):  # validation issues
                            f.write(f"{protein_info[0]}: {'; '.join(protein_info[1])}\n")
                        else:  # error reasons
                            f.write(f"{protein_info[0]}: {protein_info[1]}\n")
                    f.write("\n")
                    total_problematic += len(proteins)
            
            actual_failures = total_problematic
            if ignore_validation_issues:
                f.write(f"TOTAL FILTERED (validation issues): {len(filtered_sequences)}\n")
                f.write(f"TOTAL FAILED (InterProScan errors): {actual_failures}\n")
                f.write(f"TOTAL PROCESSED SUCCESSFULLY: {len(valid_sequences) - actual_failures}\n")
            else:
                f.write(f"TOTAL PROBLEMATIC PROTEINS: {total_problematic}\n")
                f.write(f"TOTAL PROCESSED SUCCESSFULLY: {total_proteins - total_problematic}\n")
        
        # Summary
        if ignore_validation_issues:
            interpro_failures = sum(len(proteins) for proteins in problematic_analysis.values())
            successful_count = len(valid_sequences) - interpro_failures
            update_status(f"FINAL SUMMARY:")
            update_status(f"Total proteins in input: {total_proteins}")
            update_status(f"Filtered out (validation issues): {len(filtered_sequences)}")
            update_status(f"Submitted to InterProScan: {len(valid_sequences)}")
            update_status(f"Successfully processed by InterProScan: {successful_count}")
            update_status(f"Failed in InterProScan: {interpro_failures}")
            update_status(f"Overall success rate: {(successful_count/total_proteins)*100:.1f}%")
            update_status(f"InterProScan success rate: {(successful_count/len(valid_sequences))*100:.1f}%")
        else:
            successful_count = total_proteins - sum(len(proteins) for proteins in problematic_analysis.values())
            update_status(f"FINAL SUMMARY:")
            update_status(f"Total proteins: {total_proteins}")
            update_status(f"Successfully processed: {successful_count}")
            update_status(f"Problematic proteins: {total_proteins - successful_count}")
            update_status(f"Success rate: {(successful_count/total_proteins)*100:.1f}%")
        
        # Print problematic protein categories (only for actual InterProScan failures)
        for category, proteins in problematic_analysis.items():
            if proteins and category != "sequence_validation_issues":  # Skip validation issues if they're being ignored
                update_status(f"  - {category.replace('_', ' ').title()}: {len(proteins)}")
        
        if ignore_validation_issues and filtered_sequences:
            update_status(f"Note: {len(filtered_sequences)} proteins were automatically filtered out due to validation issues")
        
        update_status(f"Detailed analysis saved to: {PROBLEMATIC_PROTEINS_FILE}")
        update_status("InterProScan processing complete!")

if __name__ == "__main__":
    main()