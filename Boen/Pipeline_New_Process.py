import sys
import os
import time
import blast2msa_new as bm
from decimal import Decimal
import Find_Parents as fp
from configure import database_dir, blast_file, go_term_dir

def run_blast(workdir, type):    # run blast

    seq_file = workdir + "/seq.fasta"

    if(os.path.exists(seq_file)==False or os.path.getsize(seq_file)==0):
        print("seq.fasta is not exist")
        return

    xml_file = workdir + "/blast_" + type + ".xml"
    database_file =  database_dir + "/" + type + "/sequence.fasta"

    cmd = blast_file + \
          " -query " + seq_file + \
          " -db " + database_file + \
          " -outfmt 5 -evalue 0.1 " \
          " -out " + xml_file

    os.system(cmd)

def extract_msa(workdir, type): # extract blast

    xml_file = workdir + "/blast_" + type + ".xml"
    seq_file = workdir + "/seq.fasta"
    msa_file = workdir + "/blast_" + type + ".msa"

    if(os.path.exists(xml_file)==False or os.path.getsize(xml_file)==0):
        print("blast.xml is not exist")
        return

    bm.run_extract_msa(seq_file, xml_file, msa_file)

def create_protein_list(workdir, go_dict, type):

    msa_file = workdir + "/blast_" + type + ".msa"

    template_list = []

    f = open(msa_file, "r")
    text = f.read()
    f.close()

    for line in text.splitlines():
        line = line.strip()
        if(line.startswith(">")):
            template = line.strip().split("\t")[0][1:]
            score = line.strip().split("\t")[1]
            if(template in go_dict):
                template_list.append([template, score])

    f = open(workdir + "/" + type + "_protein_list", "w")
    for template, score in template_list:
        f.write(template + " " + score + "\n")
    f.flush()
    f.close()


def read_protein_list(protein_list_file):    # read protein templates

    f = open(protein_list_file, "r")
    text = f.read()
    f.close()

    protein_list_dict = dict()

    for line in text.splitlines():
        values = line.strip().split()
        protein_list_dict[values[0]] = float(values[1])

    return protein_list_dict



def read_go(gofile):  # read GO Terms

    f = open(gofile, "rU")
    text = f.read()
    f.close()

    go_dict = dict()

    for line in text.splitlines():
        line = line.strip()
        values = line.split()
        go_dict[values[0]] = values[1].split(",")

    return go_dict



def annotate(workdir, type, obo_dict, go_dict):  # annotate GO term

    protein_list_file = workdir+"/" + type + "_protein_list"
    if(os.path.exists(protein_list_file)==False or os.path.getsize(protein_list_file)==0):
        print("protein list is not exist ! ")
        return

    protein_list_dict = read_protein_list(protein_list_file)

    term_list = []
    for protein in protein_list_dict:
        term_list.extend(go_dict[protein])
    term_list = list(set(term_list))

    result_dict = dict()
    for term in term_list:
        sum1 = 0.0
        sum2 = 0.0
        for protein in protein_list_dict:
            sum1 = sum1 + protein_list_dict[protein]
            if(term in go_dict[protein]):
                sum2 = sum2 + protein_list_dict[protein]

        result_dict[term] = sum2/sum1

    result_list = [(result_dict[term], term) for term in result_dict]
    result_list = sorted(result_list, reverse=True)

    resultfile = workdir + "/protein_Result_" + type
    f = open(resultfile, "w")
    for value, term in result_list:
        if(value>=0.01):
            f.write(term+" "+type[1]+" "+str(Decimal(value).quantize(Decimal("0.000"))) + "\n")
    f.flush()
    f.close()

    fp.find_parents_from_file(resultfile, resultfile+"_new", obo_dict)
    fp.sort_result(resultfile)


def process(workdir, obo_dict, protein_name, current_num, total_num):   # main process

    # Progress tracking
    progress_percent = (current_num * 100.0) / total_num
    print("=" * 60)
    print("PROGRESS: [{}/{}] {:.1f}% - Processing: {}".format(current_num, total_num, progress_percent, protein_name))
    print("=" * 60)

    type_list = ["MF", "BP", "CC"]
    start_time = time.time()

    for i, type in enumerate(type_list):
        type_progress = (i + 1) * 100.0 / len(type_list)
        print("  -> Step {}/3 ({:.0f}%): Processing {} for {}".format(i+1, type_progress, type, protein_name))
        
        step_start = time.time()
        
        print("     Running BLAST...")
        run_blast(workdir, type)
        
        print("     Extracting MSA...")
        extract_msa(workdir, type)

        go_term_file = go_term_dir + "/" + type + "_Term"
        go_dict = read_go(go_term_file)

        print("     Creating protein list...")
        create_protein_list(workdir, go_dict, type)
        
        print("     Annotating...")
        annotate(workdir, type, obo_dict, go_dict)
        
        step_time = time.time() - step_start
        print("     {} completed in {:.1f} seconds".format(type, step_time))

    total_time = time.time() - start_time
    print("  PROTEIN COMPLETED: {} in {:.1f} seconds".format(protein_name, total_time))
    
    # Time estimation
    if current_num > 0:
        avg_time_per_protein = total_time
        remaining_proteins = total_num - current_num
        estimated_remaining_time = remaining_proteins * avg_time_per_protein
        
        hours = int(estimated_remaining_time // 3600)
        minutes = int((estimated_remaining_time % 3600) // 60)
        
        print("  ESTIMATED TIME REMAINING: {} hours {} minutes".format(hours, minutes))
    
    print("")


if __name__ == '__main__':

    workdir = sys.argv[1]
    workdir = workdir.rstrip('/')  # Remove trailing slash
    obo_dict = fp.get_obo_dict()

    # Get list of all proteins and count them
    protein_list = os.listdir(workdir)
    total_proteins = len(protein_list)
    
    print("STARTING SAGP PIPELINE")
    print("Total proteins to process: {}".format(total_proteins))
    print("Estimated total time: {:.1f} - {:.1f} hours".format(total_proteins * 2 / 60.0, total_proteins * 5 / 60.0))
    print("")

    start_time = time.time()
    
    for i, name in enumerate(protein_list):
        current_protein_num = i + 1
        process(workdir + "/" + name + "/", obo_dict, name, current_protein_num, total_proteins)
    
    total_elapsed = time.time() - start_time
    hours = int(total_elapsed // 3600)
    minutes = int((total_elapsed % 3600) // 60)
    
    print("=" * 60)
    print("SAGP PIPELINE COMPLETED!")
    print("Total time: {} hours {} minutes".format(hours, minutes))
    print("Average time per protein: {:.1f} seconds".format(total_elapsed / total_proteins))
    print("=" * 60)