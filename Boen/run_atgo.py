#!/usr/bin/env python3

import os
import sys
import glob
import shutil
from datetime import datetime
from configure import script_dir, python_dir
import zipfile

def copy_master_fasta(input_fasta, output_file):
    """
    Copy the master FASTA file and count sequences
    """
    print(f"Using master FASTA file: {input_fasta}")
    
    if not os.path.exists(input_fasta):
        raise FileNotFoundError(f"Master FASTA file not found: {input_fasta}")
    
    # Copy the file
    shutil.copy2(input_fasta, output_file)
    
    # Count sequences
    with open(output_file, 'r') as f:
        content = f.read()
        sequence_count = content.count('>')
    
    print(f"Master FASTA copied with {sequence_count} sequences")
    return sequence_count

def split_sequence(workdir):
    """
    Split one sequence file as multiple sequence files
    Enhanced with better error handling and progress tracking
    """
    seq_file = workdir + "/seq.fasta"
    name_file = workdir + "/name_list"
    
    if not os.path.exists(seq_file):
        raise FileNotFoundError(f"seq.fasta not found in {workdir}")
    
    print("Splitting sequences into individual files...")
    
    with open(seq_file, "r") as f:
        text = f.read()
    
    sequence_dict = {}
    current_name = None
    current_seq = []
    
    for line in text.splitlines():
        line = line.strip()
        if line.startswith(">"):
            # Save previous sequence if exists
            if current_name and current_seq:
                sequence_dict[current_name] = ''.join(current_seq)
            # Start new sequence
            current_name = line[1:]  # Remove '>'
            current_seq = []
        elif line and current_name:
            current_seq.append(line)
    
    # Don't forget the last sequence
    if current_name and current_seq:
        sequence_dict[current_name] = ''.join(current_seq)
    
    print(f"Found {len(sequence_dict)} sequences to process")
    
    result_dir = workdir + "/SAGP/"
    if os.path.exists(result_dir):
        shutil.rmtree(result_dir)
    os.makedirs(result_dir)
    
    with open(name_file, "w") as f1:
        for i, (name, sequence) in enumerate(sequence_dict.items(), 1):
            if i % 1000 == 0:
                print(f"Processed {i}/{len(sequence_dict)} sequences")
            
            sub_dir = result_dir + "/" + name + "/"
            os.makedirs(sub_dir)
            
            with open(sub_dir + "/seq.fasta", "w") as f:
                f.write(">" + name + "\n" + sequence + "\n")
            
            f1.write(name + "\n")
    
    print(f"Split complete: {len(sequence_dict)} individual sequence files created")

def run_pipeline_step(cmd, step_name):
    """
    Run a pipeline step with error checking
    """
    print(f"\n=== Running {step_name} ===")
    print(f"Command: {cmd}")
    
    result = os.system(cmd)
    if result != 0:
        print(f"ERROR: {step_name} failed with exit code {result}")
        return False
    else:
        print(f"SUCCESS: {step_name} completed")
        return True

def all_process(workdir):
    """
    Enhanced processing pipeline with better error handling
    """
    print(f"\n=== Starting ATGO Processing Pipeline ===")
    print(f"Working directory: {workdir}")
    print(f"Timestamp: {datetime.now()}")
    
    # Step 1: Split sequences
    try:
        split_sequence(workdir)
    except Exception as e:
        print(f"ERROR in sequence splitting: {e}")
        return False
    
    # Step 2: Pipeline New Process
    cmd = f"python2 {script_dir}/Pipeline_New_Process.py {workdir}/SAGP/"
    if not run_pipeline_step(cmd, "Pipeline New Process"):
        return False
    
    # Step 3: Feature Extraction
    cmd = f"python {script_dir}/Feature_Extraction.py {workdir}"
    if not run_pipeline_step(cmd, "Feature Extraction"):
        return False
    
    # Step 4: Run Load Model
    cmd = f"{python_dir} {script_dir}/Run_Load_Model.py {workdir}/ATGO/"
    if not run_pipeline_step(cmd, "Load Model"):
        return False
    
    # Step 5: Get Average Results
    cmd = f"python {script_dir}/Get_Average_result_from_Network.py {workdir}/ATGO/"
    if not run_pipeline_step(cmd, "Average Results"):
        return False
    
    # Step 6: Clean and Combine
    atgo_plus_dir = workdir + "/ATGO_PLUS/"
    if os.path.exists(atgo_plus_dir):
        shutil.rmtree(atgo_plus_dir)
    
    cmd = f"python2 {script_dir}/Combine_result.py {workdir}"
    if not run_pipeline_step(cmd, "Combine Results"):
        return False
    
    print(f"\n=== ATGO Processing Complete ===")
    print(f"Results should be available in: {workdir}/ATGO_PLUS/")
    return True

def create_results_summary(workdir):
    """
    Create a summary of results
    """
    results_dir = workdir + "/ATGO_PLUS/"
    if not os.path.exists(results_dir):
        print("No results directory found")
        return
    
    print(f"\n=== Results Summary ===")
    print(f"Results location: {results_dir}")
    
    # List all files in results directory
    for root, dirs, files in os.walk(results_dir):
        level = root.replace(results_dir, '').count(os.sep)
        indent = ' ' * 2 * level
        print(f"{indent}{os.path.basename(root)}/")
        subindent = ' ' * 2 * (level + 1)
        for file in files:
            file_path = os.path.join(root, file)
            file_size = os.path.getsize(file_path)
            print(f"{subindent}{file} ({file_size} bytes)")

def main():
    if len(sys.argv) != 3:
        print("Usage: python atgo_batch_processor.py <master_fasta_file> <output_directory>")
        print("Example: python atgo_batch_processor.py /data/summer2020/Boen/benchmark_testing_sequences.fasta /tmp/atgo_results")
        sys.exit(1)
    
    input_fasta = sys.argv[1]
    output_dir = sys.argv[2]
    
    # Validate input file
    if not os.path.exists(input_fasta):
        print(f"ERROR: Input FASTA file does not exist: {input_fasta}")
        sys.exit(1)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"ATGO Processor Starting...")
    print(f"Input FASTA file: {input_fasta}")
    print(f"Output directory: {output_dir}")
    
    try:
        # Step 1: Copy master FASTA file to working directory
        master_fasta = os.path.join(output_dir, "seq.fasta")
        sequence_count = copy_master_fasta(input_fasta, master_fasta)
        
        # Step 2: Run ATGO pipeline
        success = all_process(output_dir)
        
        if success:
            # Step 3: Create summary
            create_results_summary(output_dir)
            print(f"\n=== PROCESSING COMPLETE ===")
            print(f"Processed {sequence_count} protein sequences")
            print(f"Results available in: {output_dir}/ATGO_PLUS/")
        else:
            print(f"\n=== PROCESSING FAILED ===")
            print("Check error messages above for details")
            sys.exit(1)
            
    except Exception as e:
        print(f"FATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()