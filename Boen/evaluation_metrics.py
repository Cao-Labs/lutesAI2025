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
                if GOtype == "BP":
                    if tem[0] not in self.BPTrueGO:
                        self.BPTrueGO[tem[0]] = []
                    self.BPTrueGO[tem[0]].append(tem[1])
                elif GOtype == "CC":
                    if tem[0] not in self.CCTrueGO:
                        self.CCTrueGO[tem[0]] = []
                    self.CCTrueGO[tem[0]].append(tem[1])
                elif GOtype == "MF":
                    if tem[0] not in self.MFTrueGO:
                        self.MFTrueGO[tem[0]] = []
                    self.MFTrueGO[tem[0]].append(tem[1])
                # anyhow, we will add GO terms to all dictionary
                if tem[0] not in self.AllTrueGO:
                    self.AllTrueGO[tem[0]] = []
                self.AllTrueGO[tem[0]].append(tem[1])

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
        if CategoryBased.lower() == "mf" or CategoryBased.lower() == "molecular_function":
            CategoryBased = "molecular_function"
        elif CategoryBased.lower() == "bp" or CategoryBased.lower() == "biological_process":
            CategoryBased = "biological_process"
        elif CategoryBased.lower() == "cc" or CategoryBased.lower() == "cellular_component":
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
                        #print "Skip this GO term"+str(line)
                        #print self.category
                        #print thecategory
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
                #print sorted_x[i]
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
            #print SameValueGO

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
        print("This script is going to do analysis of GO terms. We use GeneOntologyTree class to compare similarity of GO terms.")
        print("We use CAFA official benchmark true GO terms folder (leafonly_BPO_unique.txt, leafonly_CCO_unique.txt, leafonly_MFO_unique.txt), and your CAFA predictions folder(like CaoLab4_1_273057.txt), output folder to save results")

        showExample()
        sys.exit(0)

    goTreePath = sys.argv[1]
    dir_truePath = sys.argv[2]
    dir_predictPath = sys.argv[3]
    dir_output = sys.argv[4]

    if not os.path.exists(dir_output):
        os.makedirs(dir_output)

    #TrueGOExample = TrueProteinFunction(sys.argv[2],TestMode=0)
    #print(TrueGOExample.GetGroundTruth("BP"))

    #PredictedGOExample = PredictedProteinFunction(sys.argv[1], sys.argv[3], TestMode=0, CategoryBased = "bp")        # you need to mention where is the geneonlotyTree

    #print(PredictedGOExample.GetGOFromTarget("T992870001312"))
    #PredictedGOExample.GetPredictedGO_threshold(0.6)
    #mytop10 = PredictedGOExample.GetPredictedGO_topn(3)
    #print mytop10['T992870001312']

    #TrueBPPath = dir_truePath + "/leafonly_BPO_unique.txt"
    #TrueCCPath = dir_truePath + "/leafonly_CCO_unique.txt"
    #TrueMFPath = dir_truePath + "/leafonly_MFO_unique.txt"

    GOTree = GeneOntologyTree(goTreePath,TestMode=0)
    # 1. analyze BP, CC, MF true and predicted
    TrueGO = TrueProteinFunction(dir_truePath)
    PredictedGO_BP = PredictedProteinFunction(goTreePath, dir_predictPath, TestMode=0, CategoryBased = "BP")
    ListedTrue_BP = TrueGO.GetGroundTruth('BP')              # only keep BP category
    PredictedGO_MF = PredictedProteinFunction(goTreePath, dir_predictPath, TestMode=0, CategoryBased = "MF")
    ListedTrue_MF = TrueGO.GetGroundTruth('MF')              # only keep MF category
    PredictedGO_CC = PredictedProteinFunction(goTreePath, dir_predictPath, TestMode=0, CategoryBased = "CC")
    ListedTrue_CC = TrueGO.GetGroundTruth('CC')              # only keep CC category
    PredictedGO_ALL = PredictedProteinFunction(goTreePath, dir_predictPath, TestMode=0, CategoryBased = "ALL")
    ListedTrue_ALL = TrueGO.GetGroundTruth('ALL')              #  keep all category

    fhThresholdBP = open(dir_output+"/Threshold_BP.txt",'w')
    fhThresholdMF = open(dir_output+"/Threshold_MF.txt",'w')
    fhThresholdCC = open(dir_output+"/Threshold_CC.txt",'w')
    fhThresholdALL = open(dir_output+"/Threshold_ALL.txt",'w')

    threshBP = []
    threshMF = []
    threshCC = []
    threshALL = []
    # 2 threshold based analysis
    for thres in [x*0.01 for x in range(0,101)]:
        #print thres
        overallPreBP = 0.0

        overallPreMF = 0.0

        overallPreCC = 0.0

        overallPreALL = 0.0

        index_BP = 0
        index_CC = 0
        index_MF = 0
        index_ALL = 0
        Threshold_predicted_BP = PredictedGO_BP.GetPredictedGO_threshold(thres)
        Threshold_predicted_MF = PredictedGO_MF.GetPredictedGO_threshold(thres)
        Threshold_predicted_CC = PredictedGO_CC.GetPredictedGO_threshold(thres)
        Threshold_predicted_ALL = PredictedGO_ALL.GetPredictedGO_threshold(thres)

        for targetname in ListedTrue_ALL:
            #print targetname,
            #print ListedTrue_BP[targetname],

            if targetname in Threshold_predicted_BP:
                if targetname in ListedTrue_BP and len(ListedTrue_BP[targetname])>0:
                    precisionBP = GOTree.MaxSimilarity(PredictionSet=Threshold_predicted_BP[targetname],TrueSet=ListedTrue_BP[targetname])
                    if float(precisionBP) >= 0:
                        #fhThresholdBP.write(str(targetname)+"|True:"+str(ListedTrue_BP[targetname]) + "|Predicted:"+str(Threshold_predicted_BP[targetname])+"|Precition and Recall"+str(precisionBP)+","+str(recallBP)+"\n")
                        index_BP+=1
                        overallPreBP+=float(precisionBP)


            else:
                print("Warning, missing target for BP: " + str(targetname))

            if targetname in Threshold_predicted_MF:
                if targetname in ListedTrue_MF and len(ListedTrue_MF[targetname])>0:
                    precisionMF = GOTree.MaxSimilarity(PredictionSet=Threshold_predicted_MF[targetname],TrueSet=ListedTrue_MF[targetname])
                    if float(precisionMF) >= 0  :
                        #fhThresholdMF.write(str(targetname)+"|True:"+str(ListedTrue_MF[targetname]) + "|Predicted:"+str(Threshold_predicted_MF[targetname])+"|Precition and Recall"+str(precisionMF)+","+str(recallMF)+"\n")
                        index_MF+=1
                        overallPreMF+=float(precisionMF)

            else:
                print("Warning, missing target for MF: " + str(targetname))

            if targetname in Threshold_predicted_CC:
                if targetname in ListedTrue_CC and len(ListedTrue_CC[targetname])>0:
                    precisionCC = GOTree.MaxSimilarity(PredictionSet=Threshold_predicted_CC[targetname],TrueSet=ListedTrue_CC[targetname])
                    if float(precisionCC) >= 0  :
                        #fhThresholdCC.write(str(targetname)+"|True:"+str(ListedTrue_CC[targetname]) + "|Predicted:"+str(Threshold_predicted_CC[targetname])+"|Precition and Recall"+str(precisionCC)+","+str(recallCC)+"\n")
                        index_CC+=1
                        overallPreCC+=float(precisionCC)

            else:
                print("Warning, missing target for CC: " + str(targetname))

            if targetname in Threshold_predicted_ALL:
                precisionALL= GOTree.MaxSimilarity(PredictionSet=Threshold_predicted_ALL[targetname],TrueSet=ListedTrue_ALL[targetname])
                if float(precisionALL)>=0 :
                    #fhThresholdALL.write(str(targetname)+"|True:"+str(ListedTrue_ALL[targetname]) + "|Predicted:"+str(Threshold_predicted_ALL[targetname])+"|Precition and Recall"+str(precisionALL)+","+str(recallALL)+"\n")
                    index_ALL+=1
                    overallPreALL+=float(precisionALL)


            else:
                print("Warning, missing target for ALL: " + str(targetname))

        if index_BP!=0:
            overallPreBP/=index_BP

        if index_MF!=0:
            overallPreMF/=index_MF

        if index_CC!=0:
            overallPreCC/=index_CC

        if index_ALL!=0:
            overallPreALL/=index_ALL


        threshBP.append((thres,overallPreBP))
        threshMF.append((thres,overallPreMF))
        threshCC.append((thres,overallPreCC))
        threshALL.append((thres,overallPreALL))
        #print "For threshold" + str(thres)+" , your average model precision and recall is "+str(overallPreBP)+" and "+str(overallRecallBP)
        print("For threshold" + str(thres) + ", we evaluated BP, MF, CC, and ALL the following number: " + str(index_BP) + "," + str(index_MF) + "," + str(index_CC) + "," + str(index_ALL))

        #sys.exit(0)

    for each in threshBP:
        fhThresholdBP.write(str(each[0]) + "\t" + str(each[1])  + "\n")
    for each in threshMF:
        fhThresholdMF.write(str(each[0]) + "\t" + str(each[1]) + "\n")
    for each in threshCC:
        fhThresholdCC.write(str(each[0]) + "\t" + str(each[1]) + "\n")
    for each in threshALL:
        fhThresholdALL.write(str(each[0]) + "\t" + str(each[1])  + "\n")

    fhThresholdBP.close()
    fhThresholdMF.close()
    fhThresholdCC.close()
    fhThresholdALL.close()





    fhTopBP = open(dir_output+"/Topn_BP.txt",'w')
    fhTopMF = open(dir_output+"/Topn_MF.txt",'w')
    fhTopCC = open(dir_output+"/Topn_CC.txt",'w')
    fhTopALL = open(dir_output+"/Topn_ALL.txt",'w')

    topnBP = []
    topnMF = []
    topnCC = []
    topnALL = []
    # 3 topn based analysis
    for topn in [x for x in range(1,21)]:
        #print thres
        overallPreBP = 0.0

        overallPreMF = 0.0

        overallPreCC = 0.0

        overallPreALL = 0.0

        index_BP = 0
        index_CC = 0
        index_MF = 0
        index_ALL = 0
        Top_predicted_BP = PredictedGO_BP.GetPredictedGO_topn(topn)
        Top_predicted_MF = PredictedGO_MF.GetPredictedGO_topn(topn)
        Top_predicted_CC = PredictedGO_CC.GetPredictedGO_topn(topn)
        Top_predicted_ALL = PredictedGO_ALL.GetPredictedGO_topn(topn)

        for targetname in ListedTrue_ALL:
            #print targetname,
            #print ListedTrue_BP[targetname],
            if targetname in Top_predicted_BP:
                if targetname in ListedTrue_BP:
                    precisionBP = GOTree.MaxSimilarity(PredictionSet=Top_predicted_BP[targetname],TrueSet=ListedTrue_BP[targetname])
                    if float(precisionBP) >= 0  :
                        #fhTopBP.write(str(targetname)+"|True:"+str(ListedTrue_BP[targetname]) + "|Predicted:"+str(Top_predicted_BP[targetname])+"|Precition and Recall"+str(precisionBP)+","+str(recallBP)+"\n")
                        index_BP+=1
                        overallPreBP+=float(precisionBP)

            else:
                print("Warning, missing target for BP: " + str(targetname))

            if targetname in Top_predicted_MF:
                if targetname in ListedTrue_MF:
                    precisionMF = GOTree.MaxSimilarity(PredictionSet=Top_predicted_MF[targetname],TrueSet=ListedTrue_MF[targetname])
                    if float(precisionMF) >= 0 :
                        #fhTopMF.write(str(targetname)+"|True:"+str(ListedTrue_MF[targetname]) + "|Predicted:"+str(Top_predicted_MF[targetname])+"|Precition and Recall"+str(precisionMF)+","+str(recallMF)+"\n")
                        index_MF+=1
                        overallPreMF+=float(precisionMF)

            else:
                print("Warning, missing target for MF: " + str(targetname))

            if targetname in Top_predicted_CC:
                if targetname in ListedTrue_CC:
                    precisionCC = GOTree.MaxSimilarity(PredictionSet=Top_predicted_CC[targetname],TrueSet=ListedTrue_CC[targetname])
                    if float(precisionCC) >= 0  :
                        #fhTopCC.write(str(targetname)+"|True:"+str(ListedTrue_CC[targetname]) + "|Predicted:"+str(Top_predicted_CC[targetname])+"|Precition and Recall"+str(precisionCC)+","+str(recallCC)+"\n")
                        index_CC+=1
                        overallPreCC+=float(precisionCC)

            else:
                print("Warning, missing target for CC: " + str(targetname))

            if targetname in Top_predicted_ALL:
                precisionALL = GOTree.MaxSimilarity(PredictionSet=Top_predicted_ALL[targetname],TrueSet=ListedTrue_ALL[targetname])
                if float(precisionALL) >= 0 :
                    #fhTopALL.write(str(targetname)+"|True:"+str(ListedTrue_ALL[targetname]) + "|Predicted:"+str(Top_predicted_ALL[targetname])+"|Precition and Recall"+str(precisionALL)+","+str(recallALL)+"\n")
                    index_ALL+=1
                    overallPreALL+=float(precisionALL)

            else:
                print("Warning, missing target for ALL: " + str(targetname))

        if index_BP!=0:
            overallPreBP/=index_BP

        if index_MF!=0:
            overallPreMF/=index_MF

        if index_CC!=0:
            overallPreCC/=index_CC

        if index_ALL!=0:
            overallPreALL/=index_ALL


        topnBP.append((topn,overallPreBP))
        topnMF.append((topn,overallPreMF))
        topnCC.append((topn,overallPreCC))
        topnALL.append((topn,overallPreALL))
        #print "For Top" + str(thres)+" , your average model precision and recall is "+str(overallPreBP)+" and "+str(overallRecallBP)
        print("For Top " + str(topn) + ", we evaluated BP, MF, CC, and ALL the following number: " + str(index_BP) + "," + str(index_MF) + "," + str(index_CC) + "," + str(index_ALL))

        #sys.exit(0)

    for each in topnBP:
        fhTopBP.write(str(each[0]) + "\t" + str(each[1]) + "\n")
    for each in topnMF:
        fhTopMF.write(str(each[0]) + "\t" + str(each[1]) + "\n")
    for each in topnCC:
        fhTopCC.write(str(each[0]) + "\t" + str(each[1])  + "\n")
    for each in topnALL:
        fhTopALL.write(str(each[0]) + "\t" + str(each[1]) + "\n")

    fhTopBP.close()
    fhTopMF.close()
    fhTopCC.close()
    fhTopALL.close()


    #sys.exit(0)
    #GOTree = GeneOntologyTree(sys.argv[1],TestMode=1)

    #print(GOTree.GetGONameSpace('GO:0000002'))
    #print(GOTree.GetGONameSpace('GO:0000009'))
    #sys.exit(0)
    #print GOTree.GOSimilarity(123,343)
    #myTrueSet= ['GO:0000001','GO:0000010']
    #myPreSet= ['GO:0000011','GO:0000010','GO:0000006']
    #print(GOTree.GOSetsWithoutPropagate(PredictionSet=myPreSet,TrueSet=myTrueSet))    # make sure the first parameter is prediction, second is true
    #print(GOTree.MaxSimilarity(PredictionSet=myPreSet,TrueSet=myTrueSet))    # make sure the first parameter is prediction, second is true
    #GOTree._displayTree()





def showExample():
    print("python "+sys.argv[0] + " ../data/gene_ontology_edit.obo.2016-06-01 ../data/Selected10Data/groundtruth ../data/SelectedCaoLabPredictions/ ../result/GOAnalysisResult_CaoLab3_Maxsimilarity")

if __name__ == '__main__':
    main()
