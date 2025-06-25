import sys
import os
import os.path
from os.path import isfile, join
from os import listdir
import operator

# ============================================================================
# GENE ONTOLOGY TREE CLASS
# ============================================================================
class GeneOntologyTree:
    """
    Author: Dr. Renzhi Cao
    Written: 1/1/2020
    Any questions, please email: caora@plu.edu

    A gene ontology tree class for calculating similarity of GO terms.
    This version reads the OBO file from a specified path.
    """
    def __init__(self, pathOfGOTree, TestMode = 1):
        self.GOParent = dict()
        self.GOSpace = dict()
        self.MFroot = "GO:0003674"
        self.BProot = "GO:0008150"
        self.CCroot = "GO:0005575"
        self.TestMode = TestMode
        self.PrintMessage("Now loading a GO tree from " + str(pathOfGOTree))
        self._loadTree(pathOfGOTree)

    def PrintMessage(self,Mess):
        if self.TestMode == 1:
            print(Mess)

    def _loadTree(self, pathOfGOTree):
        """ This function loads the tree structure from an OBO file path. """
        if not os.path.isfile(pathOfGOTree):
            print(f"Error: Gene Ontology file not found at {pathOfGOTree}")
            sys.exit(1)
            
        preGO = "NULL"
        parents = []
        NameSpace = None
        with open(pathOfGOTree,'r') as fileHandle:
            for lines in fileHandle:
                tem = lines.strip().split()
                if len(tem)<2:
                    continue
                if tem[0] == "id:" and "GO:" in tem[1]:
                    if preGO!="NULL":
                        self.GOParent[preGO] = parents
                        if NameSpace is not None and preGO not in self.GOSpace:
                            self.GOSpace[preGO] = NameSpace
                    preGO = tem[1]
                    parents = []
                    NameSpace = None
                elif tem[0] == "is_a:":
                    parents.append(tem[1])
                elif tem[0] == "is_obsolete:":
                    preGO = "NULL"
                    parents = []
                elif tem[0] == "namespace:":
                    NameSpace = tem[1]
        if preGO!="NULL":
            self.GOParent[preGO] = parents
            if NameSpace and preGO not in self.GOSpace:
                 self.GOSpace[preGO] = NameSpace

    def GetGONameSpace(self, GO):
        return self.GOSpace.get(GO)

    def GOSimilarity(self, GO1, GO2):
        if GO1 not in self.GOSpace or GO2 not in self.GOSpace or self.GOSpace.get(GO1) != self.GOSpace.get(GO2):
            return 0

        path1 = self._get_path_to_root(GO1)
        path2 = self._get_path_to_root(GO2)
        
        common_ancestors = set(path1) & set(path2)
        if not common_ancestors:
            return 0
        
        deepest_common_ancestor = max(common_ancestors, key=lambda go: len(self._get_path_to_root(go)))
        
        len_path1 = len(path1)
        len_path2 = len(path2)
        len_dca = len(self._get_path_to_root(deepest_common_ancestor))

        return (2 * len_dca) / (len_path1 + len_path2) if (len_path1 + len_path2) > 0 else 0

    def _get_path_to_root(self, go_term):
        """ Helper function to get all parents up to the root. """
        if go_term not in self.GOParent:
             return [go_term]
        path = {go_term}
        to_visit = list(self.GOParent.get(go_term, []))
        while to_visit:
            current = to_visit.pop(0)
            if current not in path:
                path.add(current)
                to_visit.extend(self.GOParent.get(current, []))
        return list(path)

    def MaxSimilarity(self, PredictionSet = [], TrueSet = []):
        if not PredictionSet or not TrueSet:
            return 0.0
        
        maxSim = 0.0
        for p_go in PredictionSet:
            for t_go in TrueSet:
                sim = self.GOSimilarity(p_go, t_go)
                if sim > maxSim:
                    maxSim = sim
        return maxSim

# ============================================================================
# TRUE PROTEIN FUNCTION CLASS
# ============================================================================
class TrueProteinFunction:
    """ This class loads and stores true GO functions from CAFA-style ground truth files. """
    def __init__(self, pathGroundTruth, TestMode = 1 ):
        self.TestMode = TestMode
        self.AllTrueGO = {}
        self.BPTrueGO = {}
        self.MFTrueGO = {}
        self.CCTrueGO = {}
        
        if not os.path.isfile(pathGroundTruth):
            print(f"Error: Ground truth path not found at '{pathGroundTruth}'")
            sys.exit(1)

        self.loadFile(pathGroundTruth)

    def loadFile(self, go_path):
        self.PrintMessage("Loading file "+str(go_path))
        with open(go_path,"r") as fh:
            next(fh) # Skip header
            for line in fh:
                parts = line.strip().split('\t')
                if len(parts) < 2:
                    continue
                protein_id, go_terms_str = parts[0], parts[1]
                go_terms = [go for go in go_terms_str.split(';') if go]
                if protein_id not in self.AllTrueGO:
                    self.AllTrueGO[protein_id] = []
                self.AllTrueGO[protein_id].extend(go_terms)
    
    def GetGroundTruth(self, GOtype, go_tree=None):
        if go_tree and not self.BPTrueGO and self.AllTrueGO: # Segregate once if needed
            for protein, go_list in self.AllTrueGO.items():
                for go in go_list:
                    ns = go_tree.GetGONameSpace(go)
                    if ns == "biological_process":
                        if protein not in self.BPTrueGO: self.BPTrueGO[protein] = []
                        self.BPTrueGO[protein].append(go)
                    elif ns == "cellular_component":
                        if protein not in self.CCTrueGO: self.CCTrueGO[protein] = []
                        self.CCTrueGO[protein].append(go)
                    elif ns == "molecular_function":
                        if protein not in self.MFTrueGO: self.MFTrueGO[protein] = []
                        self.MFTrueGO[protein].append(go)
        
        if GOtype == "BP": return self.BPTrueGO
        if GOtype == "CC": return self.CCTrueGO
        if GOtype == "MF": return self.MFTrueGO
        return self.AllTrueGO

    def PrintMessage(self,Mess):
        if self.TestMode == 1:
            print(Mess)

# ============================================================================
# PREDICTED PROTEIN FUNCTION CLASS
# ============================================================================
class PredictedProteinFunction:
    """ This class loads and processes predicted GO functions. """
    def __init__(self, go_tree_obj, pathPredictions, TestMode = 1, CategoryBased = "ALL"):
        self.TestMode = TestMode
        self.AllPredictedGO = {}
        self.AllPredictedGORanked = {}

        if CategoryBased.lower() in ("mf", "molecular_function"): self.category = "molecular_function"
        elif CategoryBased.lower() in ("bp", "biological_process"): self.category = "biological_process"
        elif CategoryBased.lower() in ("cc", "cellular_component"): self.category = "cellular_component"
        else: self.category = "ALL"
            
        self.GOTree = go_tree_obj
        
        if not os.path.isfile(pathPredictions):
            print(f"Error: Predictions file not found at '{pathPredictions}'")
            sys.exit(1)
            
        self.loadFile(pathPredictions)
        self._RankAllGOterms()

    def loadFile(self, input_path):
        with open(input_path,'r') as fh:
            for line in fh:
                tem = line.split()
                if len(tem)<3 or tem[0] in ("AUTHOR", "MODEL", "KEYWORDS"):
                    continue
                
                protein_id, go_term, score = tem[0], tem[1], tem[2]
                if self.category != "ALL" and self.GOTree.GetGONameSpace(go_term) != self.category:
                    continue
                
                if protein_id not in self.AllPredictedGO:
                    self.AllPredictedGO[protein_id] = []
                self.AllPredictedGO[protein_id].append((go_term, score))

    def _RankAllGOterms(self):
        for target, preds in self.AllPredictedGO.items():
            temHash = {go: float(score) for go, score in preds}
            sorted_x = sorted(temHash.items(), key=operator.itemgetter(1), reverse = True)
            
            RankedGO = {}
            myrank = 1
            for go, score in sorted_x:
                RankedGO[go] = myrank
                myrank += 1
            
            self.AllPredictedGORanked[target] = list(RankedGO.items())


    def GetPredictedGO_threshold(self, threshold):
        ThreGO = {}
        for protein, preds in self.AllPredictedGO.items():
            for go_term, score in preds:
                if float(score) >= threshold:
                    if protein not in ThreGO: ThreGO[protein] = []
                    ThreGO[protein].append(go_term)
        return ThreGO

    def GetPredictedGO_topn(self, topn):
        TopnGO = {}
        for protein, ranked_preds in self.AllPredictedGORanked.items():
            # Ensure ranked_preds is sorted by rank
            sorted_ranks = sorted(ranked_preds, key=lambda x: x[1])
            for go_term, rank in sorted_ranks:
                if rank <= topn:
                    if protein not in TopnGO: TopnGO[protein] = []
                    TopnGO[protein].append(go_term)
        return TopnGO

# ============================================================================
# MAIN EXECUTION SCRIPT
# ============================================================================
def main():
    if len(sys.argv) < 5:
        print("Usage: python evaluation_script.py <path_to_obo> <path_to_ground_truth> <path_to_predictions> <output_dir>")
        print("\nExample:")
        print("python evaluation_script.py /data/shared/databases/UniProt2025/GO_June_1_2025 "
              "/data/summer2020/Boen/ground_truth_go_terms/consolidated_ground_truth.tsv "
              "/data/summer2020/Boen/hifun_predictions/predictions_for_eval.txt ./results")
        sys.exit(1)

    go_path = sys.argv[1]
    truth_path = sys.argv[2]
    predict_path = sys.argv[3]
    output_dir = sys.argv[4]

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    # --- Initialization ---
    print("--- Initializing ---")
    GOTree = GeneOntologyTree(go_path, TestMode=0)
    TrueGO = TrueProteinFunction(truth_path, TestMode=0)
    
    ListedTrue_BP = TrueGO.GetGroundTruth('BP', GOTree)
    ListedTrue_MF = TrueGO.GetGroundTruth('MF', GOTree)
    ListedTrue_CC = TrueGO.GetGroundTruth('CC', GOTree)
    ListedTrue_ALL = TrueGO.GetGroundTruth('ALL', GOTree)

    PredictedGO_BP = PredictedProteinFunction(GOTree, predict_path, TestMode=0, CategoryBased="BP")
    PredictedGO_MF = PredictedProteinFunction(GOTree, predict_path, TestMode=0, CategoryBased="MF")
    PredictedGO_CC = PredictedProteinFunction(GOTree, predict_path, TestMode=0, CategoryBased="CC")
    PredictedGO_ALL = PredictedProteinFunction(GOTree, predict_path, TestMode=0, CategoryBased="ALL")
    print("--- Initialization Complete ---")

    # --- Threshold-based Analysis ---
    print("\n--- Starting Threshold-based Analysis ---")
    with open(os.path.join(output_dir, "Threshold_BP.txt"), 'w') as fhThresholdBP, \
         open(os.path.join(output_dir, "Threshold_MF.txt"), 'w') as fhThresholdMF, \
         open(os.path.join(output_dir, "Threshold_CC.txt"), 'w') as fhThresholdCC, \
         open(os.path.join(output_dir, "Threshold_ALL.txt"), 'w') as fhThresholdALL:

        all_targets_with_truth = set(ListedTrue_ALL.keys())
        
        for thres in [x * 0.01 for x in range(0, 101)]:
            overallPreBP, overallPreMF, overallPreCC, overallPreALL = 0.0, 0.0, 0.0, 0.0
            index_BP, index_MF, index_CC, index_ALL = 0, 0, 0, 0

            Threshold_predicted_BP = PredictedGO_BP.GetPredictedGO_threshold(thres)
            Threshold_predicted_MF = PredictedGO_MF.GetPredictedGO_threshold(thres)
            Threshold_predicted_CC = PredictedGO_CC.GetPredictedGO_threshold(thres)
            Threshold_predicted_ALL = PredictedGO_ALL.GetPredictedGO_threshold(thres)
            
            eval_targets_bp = all_targets_with_truth & set(Threshold_predicted_BP.keys())
            for target in eval_targets_bp:
                if target in ListedTrue_BP:
                    precision = GOTree.MaxSimilarity(Threshold_predicted_BP[target], ListedTrue_BP[target])
                    overallPreBP += precision
                    index_BP += 1

            eval_targets_mf = all_targets_with_truth & set(Threshold_predicted_MF.keys())
            for target in eval_targets_mf:
                 if target in ListedTrue_MF:
                    precision = GOTree.MaxSimilarity(Threshold_predicted_MF[target], ListedTrue_MF[target])
                    overallPreMF += precision
                    index_MF += 1

            eval_targets_cc = all_targets_with_truth & set(Threshold_predicted_CC.keys())
            for target in eval_targets_cc:
                if target in ListedTrue_CC:
                    precision = GOTree.MaxSimilarity(Threshold_predicted_CC[target], ListedTrue_CC[target])
                    overallPreCC += precision
                    index_CC += 1
            
            eval_targets_all = all_targets_with_truth & set(Threshold_predicted_ALL.keys())
            for target in eval_targets_all:
                precision = GOTree.MaxSimilarity(Threshold_predicted_ALL[target], ListedTrue_ALL[target])
                overallPreALL += precision
                index_ALL += 1

            fhThresholdBP.write(f"{thres:.2f}\t{overallPreBP / index_BP if index_BP > 0 else 0:.4f}\n")
            fhThresholdMF.write(f"{thres:.2f}\t{overallPreMF / index_MF if index_MF > 0 else 0:.4f}\n")
            fhThresholdCC.write(f"{thres:.2f}\t{overallPreCC / index_CC if index_CC > 0 else 0:.4f}\n")
            fhThresholdALL.write(f"{thres:.2f}\t{overallPreALL / index_ALL if index_ALL > 0 else 0:.4f}\n")
            print(f"Threshold {thres:.2f}: Processed (BP/MF/CC/ALL): {index_BP}/{index_MF}/{index_CC}/{index_ALL}")

    print("--- Finished Threshold-based Analysis ---")

    # --- Top-N based Analysis ---
    print("\n--- Starting Top-N based Analysis ---")
    with open(os.path.join(output_dir, "Topn_BP.txt"), 'w') as fhTopBP, \
         open(os.path.join(output_dir, "Topn_MF.txt"), 'w') as fhTopMF, \
         open(os.path.join(output_dir, "Topn_CC.txt"), 'w') as fhTopCC, \
         open(os.path.join(output_dir, "Topn_ALL.txt"), 'w') as fhTopALL:
        
        all_targets_with_truth = set(ListedTrue_ALL.keys())

        for topn in range(1, 21):
            overallPreBP, overallPreMF, overallPreCC, overallPreALL = 0.0, 0.0, 0.0, 0.0
            index_BP, index_MF, index_CC, index_ALL = 0, 0, 0, 0
            
            Top_predicted_BP = PredictedGO_BP.GetPredictedGO_topn(topn)
            Top_predicted_MF = PredictedGO_MF.GetPredictedGO_topn(topn)
            Top_predicted_CC = PredictedGO_CC.GetPredictedGO_topn(topn)
            Top_predicted_ALL = PredictedGO_ALL.GetPredictedGO_topn(topn)

            eval_targets_bp = all_targets_with_truth & set(Top_predicted_BP.keys())
            for target in eval_targets_bp:
                if target in ListedTrue_BP:
                    precision = GOTree.MaxSimilarity(Top_predicted_BP[target], ListedTrue_BP[target])
                    overallPreBP += precision
                    index_BP += 1

            eval_targets_mf = all_targets_with_truth & set(Top_predicted_MF.keys())
            for target in eval_targets_mf:
                if target in ListedTrue_MF:
                    precision = GOTree.MaxSimilarity(Top_predicted_MF[target], ListedTrue_MF[target])
                    overallPreMF += precision
                    index_MF += 1

            eval_targets_cc = all_targets_with_truth & set(Top_predicted_CC.keys())
            for target in eval_targets_cc:
                if target in ListedTrue_CC:
                    precision = GOTree.MaxSimilarity(Top_predicted_CC[target], ListedTrue_CC[target])
                    overallPreCC += precision
                    index_CC += 1
            
            eval_targets_all = all_targets_with_truth & set(Top_predicted_ALL.keys())
            for target in eval_targets_all:
                precision = GOTree.MaxSimilarity(Top_predicted_ALL[target], ListedTrue_ALL[target])
                overallPreALL += precision
                index_ALL += 1

            fhTopBP.write(f"{topn}\t{overallPreBP / index_BP if index_BP > 0 else 0:.4f}\n")
            fhTopMF.write(f"{topn}\t{overallPreMF / index_MF if index_MF > 0 else 0:.4f}\n")
            fhTopCC.write(f"{topn}\t{overallPreCC / index_CC if index_CC > 0 else 0:.4f}\n")
            fhTopALL.write(f"{topn}\t{overallPreALL / index_ALL if index_ALL > 0 else 0:.4f}\n")
            print(f"Top-N {topn}: Processed (BP/MF/CC/ALL): {index_BP}/{index_MF}/{index_CC}/{index_ALL}")

    print("--- Finished Top-N based Analysis ---")
    print("\nEvaluation complete. Results are in the output directory.")


if __name__ == '__main__':
    main()
