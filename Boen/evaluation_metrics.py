# -*- coding: utf-8 -*-
"""
  Author: Dr. Renzhi Cao
  Written: 1/1/2020
  Modified by G_Gemini: 6/25/2025 
  Purpose: To perform a standard, propagated Gene Ontology evaluation for protein function predictions.

  This script calculates precision and recall for predicted GO terms against a ground truth set.
  It correctly handles the GO hierarchy by propagating annotations up to the root before comparison,
  which is the standard methodology for CAFA and other bioinformatics benchmarks.
"""
from GeneOntologyTree import *
import sys
import os
import os.path
from os.path import isfile, join
from os import listdir
import operator

class TrueProteinFunction:
    """
    This class loads and manages the ground truth GO annotations from standard CAFA-formatted files.
    """
    def __init__(self, pathGroundTruth, TestMode=0):
        self.AllTrueGO = dict()
        self.BPTrueGO = dict()
        self.MFTrueGO = dict()
        self.CCTrueGO = dict()
        self.TestMode = TestMode
        
        if os.path.isdir(pathGroundTruth):
            # Assumes a directory with specific files for each ontology branch
            BPPath = os.path.join(pathGroundTruth, "leafonly_BPO_unique.txt")
            CCPath = os.path.join(pathGroundTruth, "leafonly_CCO_unique.txt")
            MFPath = os.path.join(pathGroundTruth, "leafonly_MFO_unique.txt")
            if not all(os.path.isfile(p) for p in [BPPath, CCPath, MFPath]):
                print(f"Error: One or more required ground truth files not found in {pathGroundTruth}")
                sys.exit(1)
            self.loadFile(BPPath, "BP")
            self.loadFile(CCPath, "CC")
            self.loadFile(MFPath, "MF")
        else:
            if not os.path.isfile(pathGroundTruth):
                print(f"Error: Cannot load true GO terms from {pathGroundTruth}")
                sys.exit(1)
            self.loadFile(pathGroundTruth, "ALL")

    def loadFile(self, GOpath, GOtype):
        self.PrintMessage(f"Loading ground truth file: {GOpath}")
        with open(GOpath, "r") as fh:
            for line in fh:
                tem = line.strip().split()
                if len(tem) < 2:
                    self.PrintMessage(f"Warning: Skipping malformed line: {line.strip()}")
                    continue
                protein_id, go_term = tem[0], tem[1]

                # Add to specific ontology dictionary
                if GOtype == "BP":
                    if protein_id not in self.BPTrueGO: self.BPTrueGO[protein_id] = []
                    self.BPTrueGO[protein_id].append(go_term)
                elif GOtype == "CC":
                    if protein_id not in self.CCTrueGO: self.CCTrueGO[protein_id] = []
                    self.CCTrueGO[protein_id].append(go_term)
                elif GOtype == "MF":
                    if protein_id not in self.MFTrueGO: self.MFTrueGO[protein_id] = []
                    self.MFTrueGO[protein_id].append(go_term)

                # Always add to the master dictionary
                if protein_id not in self.AllTrueGO: self.AllTrueGO[protein_id] = []
                self.AllTrueGO[protein_id].append(go_term)

    def GetGroundTruth(self, GOtype):
        if GOtype == "BP": return self.BPTrueGO
        if GOtype == "CC": return self.CCTrueGO
        if GOtype == "MF": return self.MFTrueGO
        return self.AllTrueGO

    def PrintMessage(self, Mess):
        if self.TestMode == 1:
            print(Mess)


class PredictedProteinFunction:
    """
    This class loads and processes predicted GO annotations, filters them by category,
    and provides methods to retrieve predictions based on confidence score or rank.
    """
    def __init__(self, pathGeneOntology, pathPredictions, TestMode=0, CategoryBased="ALL"):
        self.AllPredictedGO = {}
        self.AllPredictedGORanked = {}
        self.TestMode = TestMode
        self.GOTree = GeneOntologyTree(pathGeneOntology, TestMode=0)

        # Normalize the category name for robust filtering
        cat_lower = CategoryBased.lower()
        if cat_lower in ("mf", "molecular_function"):
            self.category = "molecular_function"
        elif cat_lower in ("bp", "biological_process"):
            self.category = "biological_process"
        elif cat_lower in ("cc", "cellular_component"):
            self.category = "cellular_component"
        else:
            if cat_lower != "all":
                print(f"Warning: Did not recognize category '{CategoryBased}'. Defaulting to 'ALL'.")
            self.category = "ALL"
        
        # Load predictions from a single file or a directory of files
        if os.path.isdir(pathPredictions):
            self.loadFolders(pathPredictions)
        else:
            self.loadFile(pathPredictions)

        self._RankAllGOterms()

    def loadFile(self, input_path):
        self.PrintMessage(f"Loading prediction file: {input_path}")
        with open(input_path, 'r') as fh:
            for line in fh:
                tem = line.strip().split()
                if len(tem) < 3: continue
                if tem[0] in ("AUTHOR", "MODEL", "KEYWORDS", "END"): continue
                
                prot_id, go_term, score = tem[0], tem[1], tem[2]

                # Filter by GO namespace if a specific category is requested
                if self.category != "ALL":
                    namespace = self.GOTree.GetGONameSpace(go_term)
                    if namespace != self.category:
                        continue
                
                if prot_id not in self.AllPredictedGO: self.AllPredictedGO[prot_id] = []
                self.AllPredictedGO[prot_id].append((go_term, score))

    def loadFolders(self, inputFolder):
        onlyfiles = [join(inputFolder, f) for f in listdir(inputFolder) if isfile(join(inputFolder, f))]
        for eachfile in onlyfiles:
            self.loadFile(eachfile)

    def _RankAllGOterms(self):
        for targetname, predictions in self.AllPredictedGO.items():
            # Use a dict to handle duplicate GO terms, keeping the highest score
            temHash = {}
            for go_term, score in predictions:
                score = float(score)
                if go_term not in temHash or score > temHash[go_term]:
                    temHash[go_term] = score
            
            # Rank the unique GO terms by score
            sorted_x = sorted(temHash.items(), key=operator.itemgetter(1), reverse=True)
            
            # Assign ranks, handling ties by giving them the same average rank
            RankedGO = {}
            SameValueGO = {}
            rank = 1
            for go_term, score in sorted_x:
                RankedGO[go_term] = rank
                if score not in SameValueGO: SameValueGO[score] = []
                SameValueGO[score].append(go_term)
                rank += 1
            
            for score, go_list in SameValueGO.items():
                if len(go_list) > 1:
                    avg_rank = sum(RankedGO[go] for go in go_list) / len(go_list)
                    for go in go_list:
                        RankedGO[go] = avg_rank
            
            self.AllPredictedGORanked[targetname] = list(RankedGO.items())

    def GetPredictedGO_threshold(self, threshold):
        ThreGO = {}
        for prot_id, predictions in self.AllPredictedGO.items():
            for go_term, score in predictions:
                if float(score) >= threshold:
                    if prot_id not in ThreGO: ThreGO[prot_id] = []
                    ThreGO[prot_id].append(go_term)
        return ThreGO

    def GetPredictedGO_topn(self, topn):
        TopnGO = {}
        if int(topn) <= 0:
            print(f"Error: Top-N must be a positive integer, not {topn}")
            sys.exit(1)
        
        for prot_id, ranked_preds in self.AllPredictedGORanked.items():
            for go_term, rank in ranked_preds:
                if rank <= topn:
                    if prot_id not in TopnGO: TopnGO[prot_id] = []
                    TopnGO[prot_id].append(go_term)
        return TopnGO

    def PrintMessage(self, Mess):
        if self.TestMode == 1:
            print(Mess)


def main():
    if len(sys.argv) < 5:
        print("Usage: python evaluation_metrics.py <path_to_obo> <path_to_ground_truth> <path_to_predictions> <path_to_output_dir>")
        print("This script performs a propagated evaluation of GO terms using precision and recall.")
        showExample()
        sys.exit(0)

    goTreePath = sys.argv[1]
    dir_truePath = sys.argv[2]
    dir_predictPath = sys.argv[3]
    dir_output = sys.argv[4]

    if not os.path.exists(dir_output):
        os.makedirs(dir_output)

    # The GeneOntologyTree is instantiated once and provides the evaluation logic
    GOTree = GeneOntologyTree(goTreePath, TestMode=0)

    # Load ground truth and predictions for each category
    TrueGO = TrueProteinFunction(dir_truePath)
    ListedTrue_BP = TrueGO.GetGroundTruth('BP')
    ListedTrue_MF = TrueGO.GetGroundTruth('MF')
    ListedTrue_CC = TrueGO.GetGroundTruth('CC')
    ListedTrue_ALL = TrueGO.GetGroundTruth('ALL')
    
    PredictedGO_BP = PredictedProteinFunction(goTreePath, dir_predictPath, CategoryBased="BP")
    PredictedGO_MF = PredictedProteinFunction(goTreePath, dir_predictPath, CategoryBased="MF")
    PredictedGO_CC = PredictedProteinFunction(goTreePath, dir_predictPath, CategoryBased="CC")
    PredictedGO_ALL = PredictedProteinFunction(goTreePath, dir_predictPath, CategoryBased="ALL")

    # --- Threshold-based Propagated Analysis ---
    print("\n--- Starting Threshold-based Propagated Evaluation ---")
    output_files_thresh = {
        "BP": open(os.path.join(dir_output, "Threshold_BP_propagated.txt"), 'w'),
        "MF": open(os.path.join(dir_output, "Threshold_MF_propagated.txt"), 'w'),
        "CC": open(os.path.join(dir_output, "Threshold_CC_propagated.txt"), 'w'),
        "ALL": open(os.path.join(dir_output, "Threshold_ALL_propagated.txt"), 'w')
    }
    for f in output_files_thresh.values():
        f.write("Threshold\tPrecision\tRecall\n")

    for thres in [x * 0.01 for x in range(0, 101)]:
        metrics = {cat: {'p_sum': 0.0, 'r_sum': 0.0, 'count': 0} for cat in ["BP", "MF", "CC", "ALL"]}
        
        predictions_thresh = {
            "BP": PredictedGO_BP.GetPredictedGO_threshold(thres),
            "MF": PredictedGO_MF.GetPredictedGO_threshold(thres),
            "CC": PredictedGO_CC.GetPredictedGO_threshold(thres),
            "ALL": PredictedGO_ALL.GetPredictedGO_threshold(thres)
        }
        
        true_sets = {"BP": ListedTrue_BP, "MF": ListedTrue_MF, "CC": ListedTrue_CC, "ALL": ListedTrue_ALL}

        for target in true_sets["ALL"]:
            for cat in ["BP", "MF", "CC", "ALL"]:
                if target in predictions_thresh[cat] and target in true_sets[cat]:
                    precision, recall = GOTree.GOSetsPropagate(
                        predictions_thresh[cat][target],
                        true_sets[cat][target]
                    )
                    if precision >= 0 and recall >= 0:
                        metrics[cat]['p_sum'] += precision
                        metrics[cat]['r_sum'] += recall
                        metrics[cat]['count'] += 1
        
        print(f"Threshold {thres:.2f} | Evaluated (BP,MF,CC,ALL): {metrics['BP']['count']},{metrics['MF']['count']},{metrics['CC']['count']},{metrics['ALL']['count']}")

        for cat, data in metrics.items():
            if data['count'] > 0:
                avg_p = data['p_sum'] / data['count']
                avg_r = data['r_sum'] / data['count']
                output_files_thresh[cat].write(f"{thres:.2f}\t{avg_p:.5f}\t{avg_r:.5f}\n")

    for f in output_files_thresh.values(): f.close()

    # --- Top-N based Propagated Analysis ---
    print("\n--- Starting Top-N-based Propagated Evaluation ---")
    output_files_topn = {
        "BP": open(os.path.join(dir_output, "TopN_BP_propagated.txt"), 'w'),
        "MF": open(os.path.join(dir_output, "TopN_MF_propagated.txt"), 'w'),
        "CC": open(os.path.join(dir_output, "TopN_CC_propagated.txt"), 'w'),
        "ALL": open(os.path.join(dir_output, "TopN_ALL_propagated.txt"), 'w')
    }
    for f in output_files_topn.values():
        f.write("TopN\tPrecision\tRecall\n")

    for topn in range(1, 21):
        metrics = {cat: {'p_sum': 0.0, 'r_sum': 0.0, 'count': 0} for cat in ["BP", "MF", "CC", "ALL"]}
        
        predictions_topn = {
            "BP": PredictedGO_BP.GetPredictedGO_topn(topn),
            "MF": PredictedGO_MF.GetPredictedGO_topn(topn),
            "CC": PredictedGO_CC.GetPredictedGO_topn(topn),
            "ALL": PredictedGO_ALL.GetPredictedGO_topn(topn)
        }
        
        true_sets = {"BP": ListedTrue_BP, "MF": ListedTrue_MF, "CC": ListedTrue_CC, "ALL": ListedTrue_ALL}
        
        for target in true_sets["ALL"]:
            for cat in ["BP", "MF", "CC", "ALL"]:
                if target in predictions_topn[cat] and target in true_sets[cat]:
                    precision, recall = GOTree.GOSetsPropagate(
                        predictions_topn[cat][target],
                        true_sets[cat][target]
                    )
                    if precision >= 0 and recall >= 0:
                        metrics[cat]['p_sum'] += precision
                        metrics[cat]['r_sum'] += recall
                        metrics[cat]['count'] += 1

        print(f"Top {topn} | Evaluated (BP,MF,CC,ALL): {metrics['BP']['count']},{metrics['MF']['count']},{metrics['CC']['count']},{metrics['ALL']['count']}")

        for cat, data in metrics.items():
            if data['count'] > 0:
                avg_p = data['p_sum'] / data['count']
                avg_r = data['r_sum'] / data['count']
                output_files_topn[cat].write(f"{topn}\t{avg_p:.5f}\t{avg_r:.5f}\n")
    
    for f in output_files_topn.values(): f.close()

    print("\nEvaluation complete. Results saved in:", dir_output)


def showExample():
    print("Example usage:")
    print("python evaluation_metrics.py ../data/go.obo ../data/groundtruth/ ../data/predictions/ ../results/PropagatedResults")


if __name__ == '__main__':
    main()
