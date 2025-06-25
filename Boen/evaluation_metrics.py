from GeneOntologyTree import *
import sys
import os
import os.path
from os.path import isfile, join
from os import listdir
import operator

"""
  Author: Dr. Renzhi Cao
  Written: 1/1/2020
  Any questions, please email: caora@plu.edu

This simple class is going to save all true functions for each target in CAFA, we load the groundtruth from CAFA released files
"""
class TrueProteinFunction:
    AllTrueGO = dict()
    BPTrueGO = dict()
    MFTrueGO = dict()
    CCTrueGO = dict()
    def __init__(self, pathGroundTruth, TestMode = 1 ):
        self.TestMode = TestMode
        # we consider import from a folder or simply from a file
        if os.path.isdir(pathGroundTruth):  # we have three files for each category
            BPPath = pathGroundTruth+"/leafonly_BPO_unique.txt"
            CCPath = pathGroundTruth+"/leafonly_CCO_unique.txt"
            MFPath = pathGroundTruth+"/leafonly_MFO_unique.txt"
            if (not os.path.isfile(BPPath)) or (not os.path.isfile(CCPath)) or (not os.path.isfile(MFPath)):
                print("Error, cannot load the true GO terms, not existing "+str(BPPath)+" or "+str(CCPath)+" or "+str(MFPath))
                sys.exit(0)
            self.loadFile(BPPath, "BP")
            self.loadFile(CCPath, "CC")
            self.loadFile(MFPath, "MF")

        else:
            if not os.path.isfile(pathGroundTruth):
                print("Error, cannot load true GO terms from " + str(pathGroundTruth))
                sys.exit(0)
            self.loadFile(pathGroundTruth, "ALL")

    """
        This function is going to load true GO terms file
    """
    def loadFile(self, GOpath, GOtype):
        self.PrintMessage("Loading file "+str(GOpath))
        with open(GOpath,"r") as fh:
            for line in fh:
                tem = line.split()
                if len(tem)<2:
                    print("Warning, what is "+ str(line))
                    continue
                protein_id = tem[0]
                go_term = tem[1]
                if GOtype == "BP":
                    if protein_id not in self.BPTrueGO:
                        self.BPTrueGO[protein_id] = []
                    self.BPTrueGO[protein_id].append(go_term)
                elif GOtype == "CC":
                    if protein_id not in self.CCTrueGO:
                        self.CCTrueGO[protein_id] = []
                    self.CCTrueGO[protein_id].append(go_term)
                elif GOtype == "MF":
                    if protein_id not in self.MFTrueGO:
                        self.MFTrueGO[protein_id] = []
                    self.MFTrueGO[protein_id].append(go_term)
                # anyhow, we will add GO terms to all dictionary
                if protein_id not in self.AllTrueGO:
                    self.AllTrueGO[protein_id] = []
                self.AllTrueGO[protein_id].append(go_term)

    """
        This function is going to return true GO terms
    """
    def GetGroundTruth(self, GOtype):
        if GOtype == "BP":
            return self.BPTrueGO
        elif GOtype == "CC":
            return self.CCTrueGO
        elif GOtype == "MF":
            return self.MFTrueGO
        else:
            return self.AllTrueGO

    def PrintMessage(self,Mess):       # when you are testing mode, you can get these messages
        if self.TestMode == 1:
            print(Mess)


"""
This simple class is going to process predicted GO functions
"""
class PredictedProteinFunction:
    AllPredictedGO = dict()         # load all predicted GO terms. Key is target name, value is a dictionary, each value is a list. [ (GO, score), ... ]
    AllPredictedGORanked = dict()         # load all predicted GO terms. Key is target name, value is a dictionary, each value is a list. [ (GO, rank), ... ]
    def __init__(self, pathGeneOntology, pathPredictions, TestMode = 1, CategoryBased = "ALL"):          # you may need to assign BP, MF, CC to CategoryBased if you only want to consider specific category of GO terms
        self.TestMode = TestMode
        if CategoryBased.lower() in ("mf", "molecular_function"):
            CategoryBased = "molecular_function"
        elif CategoryBased.lower() in ("bp", "biological_process"):
            CategoryBased = "biological_process"
        elif CategoryBased.lower() in ("cc", "cellular_component"):
            CategoryBased = "cellular_component"
        else:
            if CategoryBased != "ALL":
                print("Warning, don't recognize "+str(CategoryBased))
                print("Set it to ALL")
            CategoryBased = "ALL"

        self.category = CategoryBased
        self.GOTree = GeneOntologyTree(pathGeneOntology,TestMode=0)
        if os.path.isdir(pathPredictions):  # we have folders for all categories
            self.loadFolders(pathPredictions)
        else:
            self.loadFile(pathPredictions)

        self._RankAllGOterms()       # we will simply process AllPredictedGO and get AllPredictedGORanked

    def loadFile(self,input):
        with open(input,'r') as fh:
            for line in fh:
                tem = line.split()
                if len(tem)<3:
                    self.PrintMessage("Skip "+str(line))
                    continue
                if tem[0] == "AUTHOR" or tem[0] == "MODEL" or tem[0] == "KEYWORDS":
                    continue
                if self.category != "ALL":     # you only want to consider specific GO category
                    thecategory = self.GOTree.GetGONameSpace(tem[1])
                    if thecategory!= self.category:
                        continue
                if tem[0] not in self.AllPredictedGO:
                    self.AllPredictedGO[tem[0]] = []
                self.AllPredictedGO[tem[0]].append((tem[1],tem[2]))

    def loadFolders(self,inputFolder):
        onlyfiles = [join(inputFolder,f) for f in listdir(inputFolder) if isfile(join(inputFolder, f))]
        for eachfile in onlyfiles:
            self.loadFile(eachfile)

    def GetGOFromTarget(self, targetname):
        if targetname in self.AllPredictedGO:
            return self.AllPredictedGO[targetname]
        else:
            return None

    def PrintMessage(self,Mess):       # when you are testing mode, you can get these messages
        if self.TestMode == 1:
            print(Mess)

    def _RankAllGOterms(self):        # this function is going to rank all GO terms and create a ranked list
        for targetname in self.AllPredictedGO:
            originalGOlist = self.AllPredictedGO[targetname]            # a list : [(GO,score)...]
            temHash = dict()
            for eachpair in originalGOlist:
                if eachpair[0] not in temHash:
                    temHash[eachpair[0]] = float(eachpair[1])
            # first rank them by scores
            sorted_x = sorted(temHash.items(), key=operator.itemgetter(1), reverse = True)
            myrank = 1
            RankedGO = dict()
            SameValueGO = dict()       # key is score, value is list of GO with the same score
            for i in range(len(sorted_x)):
                RankedGO[sorted_x[i][0]] = myrank
                if sorted_x[i][1] not in SameValueGO:
                    SameValueGO[sorted_x[i][1]] = []
                SameValueGO[sorted_x[i][1]].append(sorted_x[i][0])
                myrank+=1
            # now update the rank, for the GOs with same score, give the average rank for them
            for each in SameValueGO:
                listGOs = SameValueGO[each]
                if len(listGOs)>1:
                    averageRank = 0
                    for eachGO in listGOs:
                        averageRank+=RankedGO[eachGO]
                    averageRank/=float(len(listGOs))
                    for eachGO in listGOs:
                        RankedGO[eachGO] = averageRank
            # now build the new GO list with ranking
            rankedGOlist = []
            for each in RankedGO:
                rankedGOlist.append((each,RankedGO[each]))
            self.AllPredictedGORanked[targetname] = rankedGOlist

    """
       THis function is used to analyze GO terms by selecting a threshold to filter all GO terms. Only return the dictionary and for each target, the GO terms larger or equal to threshold will be kept
    """
    def GetPredictedGO_threshold(self, threshold):
        ThreGO = dict()      # a new dictionary for results
        for each in self.AllPredictedGO:
            TargetGO = self.AllPredictedGO[each]
            for GOInfor in TargetGO:
                if float(GOInfor[1]) >= float(threshold):
                    if each not in ThreGO:
                        ThreGO[each] = []
                    ThreGO[each].append(GOInfor[0])
        return ThreGO

    """
       THis function is used to analyze GO terms by selecting top n to filter all GO terms. Only return the dictionary and for each target, the GO terms rank higher than the rank will be kept
    """
    def GetPredictedGO_topn(self, topn):
        TopnGO = dict()        # keep top n GOs
        topn = int(topn)
        if topn <= 0:
            print("Error, cannot select top n with this number n:"+str(topn))
            sys.exit(0)

        for targetname in self.AllPredictedGORanked:
            TargetGO = self.AllPredictedGORanked[targetname]
            for GOInfor in TargetGO:
                if float(GOInfor[1]) <= float(topn):
                    if targetname not in TopnGO:
                        TopnGO[targetname] = []
                    TopnGO[targetname].append(GOInfor[0])
        return TopnGO



def main():
    if len(sys.argv) < 5:
        print("This script is going to do analysis of GO terms using precision and recall.")
        print("We use CAFA official benchmark true GO terms folder (leafonly_BPO_unique.txt, leafonly_CCO_unique.txt, leafonly_MFO_unique.txt), and your CAFA predictions folder(like CaoLab4_1_273057.txt), output folder to save results")

        showExample()
        sys.exit(0)

    goTreePath = sys.argv[1]
    dir_truePath = sys.argv[2]
    dir_predictPath = sys.argv[3]
    dir_output = sys.argv[4]

    if not os.path.exists(dir_output):
        os.makedirs(dir_output)

    TrueGO = TrueProteinFunction(dir_truePath)
    PredictedGO_BP = PredictedProteinFunction(goTreePath, dir_predictPath, TestMode=0, CategoryBased="BP")
    ListedTrue_BP = TrueGO.GetGroundTruth('BP')
    PredictedGO_MF = PredictedProteinFunction(goTreePath, dir_predictPath, TestMode=0, CategoryBased="MF")
    ListedTrue_MF = TrueGO.GetGroundTruth('MF')
    PredictedGO_CC = PredictedProteinFunction(goTreePath, dir_predictPath, TestMode=0, CategoryBased="CC")
    ListedTrue_CC = TrueGO.GetGroundTruth('CC')
    PredictedGO_ALL = PredictedProteinFunction(goTreePath, dir_predictPath, TestMode=0, CategoryBased="ALL")
    ListedTrue_ALL = TrueGO.GetGroundTruth('ALL')

    fhThresholdBP = open(dir_output + "/Threshold_BP.txt", 'w')
    fhThresholdMF = open(dir_output + "/Threshold_MF.txt", 'w')
    fhThresholdCC = open(dir_output + "/Threshold_CC.txt", 'w')
    fhThresholdALL = open(dir_output + "/Threshold_ALL.txt", 'w')

    # Write headers to output files
    fhThresholdBP.write("Threshold\tPrecision\tRecall\n")
    fhThresholdMF.write("Threshold\tPrecision\tRecall\n")
    fhThresholdCC.write("Threshold\tPrecision\tRecall\n")
    fhThresholdALL.write("Threshold\tPrecision\tRecall\n")

    threshBP, threshMF, threshCC, threshALL = [], [], [], []
    
    # 2 threshold based analysis
    for thres in [x * 0.01 for x in range(0, 101)]:
        overallPrecisionBP, overallRecallBP = 0.0, 0.0
        overallPrecisionMF, overallRecallMF = 0.0, 0.0
        overallPrecisionCC, overallRecallCC = 0.0, 0.0
        overallPrecisionALL, overallRecallALL = 0.0, 0.0
        
        index_BP, index_CC, index_MF, index_ALL = 0, 0, 0, 0

        Threshold_predicted_BP = PredictedGO_BP.GetPredictedGO_threshold(thres)
        Threshold_predicted_MF = PredictedGO_MF.GetPredictedGO_threshold(thres)
        Threshold_predicted_CC = PredictedGO_CC.GetPredictedGO_threshold(thres)
        Threshold_predicted_ALL = PredictedGO_ALL.GetPredictedGO_threshold(thres)

        for targetname in ListedTrue_ALL:
            if targetname in Threshold_predicted_BP:
                if targetname in ListedTrue_BP and len(ListedTrue_BP[targetname]) > 0:
                    precision, recall = calculate_precision_recall(Threshold_predicted_BP[targetname], ListedTrue_BP[targetname])
                    overallPrecisionBP += precision
                    overallRecallBP += recall
                    index_BP += 1
            else:
                print("Warning, missing target for BP: " + str(targetname))

            if targetname in Threshold_predicted_MF:
                if targetname in ListedTrue_MF and len(ListedTrue_MF[targetname]) > 0:
                    precision, recall = calculate_precision_recall(Threshold_predicted_MF[targetname], ListedTrue_MF[targetname])
                    overallPrecisionMF += precision
                    overallRecallMF += recall
                    index_MF += 1
            else:
                print("Warning, missing target for MF: " + str(targetname))

            if targetname in Threshold_predicted_CC:
                if targetname in ListedTrue_CC and len(ListedTrue_CC[targetname]) > 0:
                    precision, recall = calculate_precision_recall(Threshold_predicted_CC[targetname], ListedTrue_CC[targetname])
                    overallPrecisionCC += precision
                    overallRecallCC += recall
                    index_CC += 1
            else:
                print("Warning, missing target for CC: " + str(targetname))

            if targetname in Threshold_predicted_ALL:
                if targetname in ListedTrue_ALL and len(ListedTrue_ALL[targetname]) > 0:
                    precision, recall = calculate_precision_recall(Threshold_predicted_ALL[targetname], ListedTrue_ALL[targetname])
                    overallPrecisionALL += precision
                    overallRecallALL += recall
                    index_ALL += 1
            else:
                print("Warning, missing target for ALL: " + str(targetname))

        if index_BP != 0:
            threshBP.append((thres, overallPrecisionBP / index_BP, overallRecallBP / index_BP))
        if index_MF != 0:
            threshMF.append((thres, overallPrecisionMF / index_MF, overallRecallMF / index_MF))
        if index_CC != 0:
            threshCC.append((thres, overallPrecisionCC / index_CC, overallRecallCC / index_CC))
        if index_ALL != 0:
            threshALL.append((thres, overallPrecisionALL / index_ALL, overallRecallALL / index_ALL))

        print("For threshold " + str(thres) + ", we evaluated BP, MF, CC, and ALL the following number: " + str(index_BP) + "," + str(index_MF) + "," + str(index_CC) + "," + str(index_ALL))

    for each in threshBP:
        fhThresholdBP.write(str(each[0]) + "\t" + str(each[1]) + "\t" + str(each[2]) + "\n")
    for each in threshMF:
        fhThresholdMF.write(str(each[0]) + "\t" + str(each[1]) + "\t" + str(each[2]) + "\n")
    for each in threshCC:
        fhThresholdCC.write(str(each[0]) + "\t" + str(each[1]) + "\t" + str(each[2]) + "\n")
    for each in threshALL:
        fhThresholdALL.write(str(each[0]) + "\t" + str(each[1]) + "\t" + str(each[2]) + "\n")

    fhThresholdBP.close()
    fhThresholdMF.close()
    fhThresholdCC.close()
    fhThresholdALL.close()

    fhTopBP = open(dir_output + "/Topn_BP.txt", 'w')
    fhTopMF = open(dir_output + "/Topn_MF.txt", 'w')
    fhTopCC = open(dir_output + "/Topn_CC.txt", 'w')
    fhTopALL = open(dir_output + "/Topn_ALL.txt", 'w')

    # Write headers to top-n files
    fhTopBP.write("TopN\tPrecision\tRecall\n")
    fhTopMF.write("TopN\tPrecision\tRecall\n")
    fhTopCC.write("TopN\tPrecision\tRecall\n")
    fhTopALL.write("TopN\tPrecision\tRecall\n")

    topnBP, topnMF, topnCC, topnALL = [], [], [], []

    # 3 topn based analysis
    for topn in [x for x in range(1, 21)]:
        overallPrecisionBP, overallRecallBP = 0.0, 0.0
        overallPrecisionMF, overallRecallMF = 0.0, 0.0
        overallPrecisionCC, overallRecallCC = 0.0, 0.0
        overallPrecisionALL, overallRecallALL = 0.0, 0.0

        index_BP, index_CC, index_MF, index_ALL = 0, 0, 0, 0

        Top_predicted_BP = PredictedGO_BP.GetPredictedGO_topn(topn)
        Top_predicted_MF = PredictedGO_MF.GetPredictedGO_topn(topn)
        Top_predicted_CC = PredictedGO_CC.GetPredictedGO_topn(topn)
        Top_predicted_ALL = PredictedGO_ALL.GetPredictedGO_topn(topn)

        for targetname in ListedTrue_ALL:
            if targetname in Top_predicted_BP:
                if targetname in ListedTrue_BP and len(ListedTrue_BP[targetname]) > 0:
                    precision, recall = calculate_precision_recall(Top_predicted_BP[targetname], ListedTrue_BP[targetname])
                    overallPrecisionBP += precision
                    overallRecallBP += recall
                    index_BP += 1
            else:
                print("Warning, missing target for BP: " + str(targetname))
            
            if targetname in Top_predicted_MF:
                if targetname in ListedTrue_MF and len(ListedTrue_MF[targetname]) > 0:
                    precision, recall = calculate_precision_recall(Top_predicted_MF[targetname], ListedTrue_MF[targetname])
                    overallPrecisionMF += precision
                    overallRecallMF += recall
                    index_MF += 1
            else:
                print("Warning, missing target for MF: " + str(targetname))

            if targetname in Top_predicted_CC:
                if targetname in ListedTrue_CC and len(ListedTrue_CC[targetname]) > 0:
                    precision, recall = calculate_precision_recall(Top_predicted_CC[targetname], ListedTrue_CC[targetname])
                    overallPrecisionCC += precision
                    overallRecallCC += recall
                    index_CC += 1
            else:
                print("Warning, missing target for CC: " + str(targetname))

            if targetname in Top_predicted_ALL:
                if targetname in ListedTrue_ALL and len(ListedTrue_ALL[targetname]) > 0:
                    precision, recall = calculate_precision_recall(Top_predicted_ALL[targetname], ListedTrue_ALL[targetname])
                    overallPrecisionALL += precision
                    overallRecallALL += recall
                    index_ALL += 1
            else:
                print("Warning, missing target for ALL: " + str(targetname))

        if index_BP != 0:
            topnBP.append((topn, overallPrecisionBP / index_BP, overallRecallBP / index_BP))
        if index_MF != 0:
            topnMF.append((topn, overallPrecisionMF / index_MF, overallRecallMF / index_MF))
        if index_CC != 0:
            topnCC.append((topn, overallPrecisionCC / index_CC, overallRecallCC / index_CC))
        if index_ALL != 0:
            topnALL.append((topn, overallPrecisionALL / index_ALL, overallRecallALL / index_ALL))
            
        print("For Top " + str(topn) + ", we evaluated BP, MF, CC, and ALL the following number: " + str(index_BP) + "," + str(index_MF) + "," + str(index_CC) + "," + str(index_ALL))

    for each in topnBP:
        fhTopBP.write(str(each[0]) + "\t" + str(each[1]) + "\t" + str(each[2]) + "\n")
    for each in topnMF:
        fhTopMF.write(str(each[0]) + "\t" + str(each[1]) + "\t" + str(each[2]) + "\n")
    for each in topnCC:
        fhTopCC.write(str(each[0]) + "\t" + str(each[1]) + "\t" + str(each[2]) + "\n")
    for each in topnALL:
        fhTopALL.write(str(each[0]) + "\t" + str(each[1]) + "\t" + str(each[2]) + "\n")

    fhTopBP.close()
    fhTopMF.close()
    fhTopCC.close()
    fhTopALL.close()

def showExample():
    print("python " + sys.argv[0] + " ../data/gene_ontology_edit.obo.2016-06-01 ../data/Selected10Data/groundtruth ../data/SelectedCaoLabPredictions/ ../result/GOAnalysisResult_PrecisionRecall")

if __name__ == '__main__':
    main()