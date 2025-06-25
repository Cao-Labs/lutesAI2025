"""
Author: Dr. Renzhi Cao
Written: 1/1/2020
Any questions, please email: caora@plu.edu

A gene ontology tree class for different methods of calculating similarity of GO terms, whether two GO terms consider as match by propagation
# the rule of propagation is learned from: https://www.biomedcentral.com/content/supplementary/1471-2105-13-S4-S14-S7.pdf
# basically, propagate all true GO terms, collecting all GO terms on the way to the root. Also propagate all predicted GO terms. Now for this two sets, calculate
# how many of them are TP (predicted correctly), FP (predicted, but wrong), FN (not predicted, but correct). Precision = TP/(TP+FP), Recall = TP/(TP+FN)
"""
import sys

class GeneOntologyTree:
# import a GO tree to your system
    def __init__(self, pathOfGOTree, TestMode = 1):
        self.GOParent = dict()
        self.GOSpace = dict()       # store GO id and namespace of this GO
        self.MFroot = "GO:0003674"
        self.BProot = "GO:0008150"
        self.CCroot = "GO:0005575"
        self.TestMode = TestMode
        self.PrintMessage("Now loading a GO tree from " + str(pathOfGOTree))
        self._loadTree(pathOfGOTree)

    def PrintMessage(self,Mess):       # when you are testing mode, you can get these messages
        if self.TestMode == 1:
            print(Mess)

    """
    Calculate the similarity between two GO terms, return -1 if one of them are not existed in the GO tree.
    Similarity = length of common path / (length of common path + length of longest leaf to the first sharing node)
    Method: from two GO node, we go to their parents by uplevel, until one of the parents are in the path of another, we continue to go up level until the root, count the number of steps.
    """
    def GOSimilarity(self, GO1, GO2):
        self.PrintMessage("Now compute the similarity of two GO terms sets based on GO tree. We may need to DFS to get the path from GO to the root.")
        if GO1 not in self.GOSpace or GO2 not in self.GOSpace:
            return -1

        if self.GOSpace[GO1] != self.GOSpace[GO2]:      # no similarity for two go terms in different tree
            return 0

        Longest2Share = 0
        CommonStep = 0

        Level1 = []
        Sets1 = dict()
        Level2 = []
        Sets2 = dict()
        Level1.append(GO1)
        Level2.append(GO2)
        Sets1[GO1] = 1
        Sets2[GO2] = 1

        StartingLevel = None

        # each node go uplevel step by step untill we find the first shared node
        while 1:
            #print Level1,Level2, Sets1, Sets2
            # check if sets1 node already in sets2
            tag = 0
            for eachGO in Level1:
                if eachGO in Sets2:
                    StartingLevel = Level1
                    tag = 1
            if tag == 1:
                break
            tag = 0
            for eachGO in Level2:
                if eachGO in Sets1:
                    StartingLevel = Level2
                    tag = 1
            if tag == 1:
                break
            Longest2Share+=1
            # update level and sets
            temLevel = []
            for eachGO in Level1:
                if (eachGO != self.MFroot) and  (eachGO != self.BProot) and (eachGO != self.CCroot) and (eachGO in self.GOParent):
                    for tGO in self.GOParent[eachGO]:
                        temLevel.append(tGO)
                        Sets1[tGO] = 1
            Level1 = temLevel
            temLevel = []
            for eachGO in Level2:
                if (eachGO != self.MFroot) and  (eachGO != self.BProot) and (eachGO != self.CCroot) and (eachGO in self.GOParent):
                    for tGO in self.GOParent[eachGO]:
                        temLevel.append(tGO)
                        Sets2[tGO] = 1
            Level2 = temLevel

            if len(Level1) == 0 or len(Level2) == 0:      # This means one of them already at the root, but another is still not, so they only share the root. I give it 1 as common step, and divide by 1 + longest
                finalSimilarity = 1.0 / (Longest2Share + 1.0)
                return finalSimilarity

        # now we find the shared level, try to get the step of common
        while 1:
            CommonStep+=1
            temLevel = []
            for eachGO in StartingLevel:
                if (eachGO != self.MFroot) and  (eachGO != self.BProot) and (eachGO != self.CCroot) and (eachGO in self.GOParent):
                    for tGO in self.GOParent[eachGO]:
                        temLevel.append(tGO)
            StartingLevel = temLevel
            if len(StartingLevel) == 0:
                break

        #print(CommonStep)
        #print(Longest2Share)
        finalSimilarity = float(CommonStep) / (CommonStep + Longest2Share)
        return finalSimilarity

    """
    This function is going to evaluate two set of GO terms and return the similarity of max
    """
    def MaxSimilarity(self, PredictionSet = [], TrueSet = []):
        maxSim = 0.0
        for preGO in PredictionSet:
            for tGO in TrueSet:
                temSim = self.GOSimilarity(preGO,tGO)
                if temSim > maxSim:
                    maxSim = temSim
        return maxSim

    """
    This function is going to evaluate two set of GO terms and return the similarity of average
    The GO not in the same tree are not considered
    """
    def AveSimilarity(self, PredictionSet = [], TrueSet = []):
        AveSim = 0.0
        index = 0
        for preGO in PredictionSet:
            for tGO in TrueSet:
                temSim = self.GOSimilarity(preGO,tGO)
                if temSim < 0:
                    continue
                AveSim+=temSim
                index+=1
        if index == 0:
            return 0
        else:
            return AveSim/index



    # this function is going to return what kind of tree of your GO terms belonging to
    def GetGONameSpace(self, GO):
        if GO in self.GOSpace:
            return self.GOSpace[GO]
        else:
            return None


    def GOSetsPropagate(self, PredictionSet, TrueSet):
        self.PrintMessage("check precision, recall, TP, TN, FN, for two sets of GO terms when propagate them to the root. One sets is predicted, and other set is true GO terms")
        #print PredictionSet, TrueSet
        if PredictionSet == None or TrueSet == None or len(PredictionSet) == 0 or len(TrueSet) == 0:
            return (0,0)        # precision and recall is 0 when one of dataset is empty
        NewPre = dict()
        NewTrue= dict()
        for each in PredictionSet:
            if "GO:" not in each:    # you forget to add GO:
                each = "GO:"+each
            if (each == self.MFroot) or (each == self.BProot) or (each == self.CCroot):   # remove the root GO terms
                continue
            if each not in self.GOParent:
                self.PrintMessage("Warning, the following predicted GO term is not in GeneOntologyTree, maybe because of version problem?"+str(each))
                #print("Warning, the following predicted GO term is not in GeneOntologyTree, maybe because of version problem?"+str(each))
                continue
            for newGO in self._propagateGO(each):
                if newGO not in NewPre:
                    NewPre[newGO] = 1
        for each in TrueSet:
            if "GO:" not in each:    # you forget to add GO:
                each = "GO:"+each
            if (each == self.MFroot) or (each == self.BProot) or (each == self.CCroot):  # remove the root GO terms
                continue
            if each not in self.GOParent:
                self.PrintMessage("Warning, the following true GO term is not in GeneOntologyTree, maybe because of version problem?"+str(each))
                continue
            for newGO in self._propagateGO(each):
                if newGO not in NewTrue:
                    NewTrue[newGO] = 1

        self.PrintMessage(str(NewPre))
        self.PrintMessage(str(NewTrue))
        # now you get clean pre and true GO sets for evaluation
        return self._CalPrecisionRecall(NewPre,NewTrue)

    # calculate precision and recall of two GO sets without propagating
    def GOSetsWithoutPropagate(self, PredictionSet, TrueSet):
        self.PrintMessage("check precision, recall, TP, TN, FN, for two sets of GO terms. PredictionSet is predicted, and other set is true GO terms")
        if PredictionSet == None or TrueSet == None or len(PredictionSet) == 0 or len(TrueSet) == 0:
            return (0,0)        # precision and recall is 0 when one of dataset is empty
        NewPre = dict()
        NewTrue= dict()
        for each in PredictionSet:
            if "GO:" not in each:    # you forget to add GO:
                each = "GO:"+each
            if (each == self.MFroot) or (each == self.BProot) or (each == self.CCroot):   # remove the root GO terms
                continue
            if each not in self.GOParent:
                self.PrintMessage("Warning, the following predicted GO term is not in GeneOntologyTree, maybe because of version problem?"+str(each))
                continue
            if each not in NewPre:
                NewPre[each] = 1
        for each in TrueSet:
            if "GO:" not in each:    # you forget to add GO:
                each = "GO:"+each
            if (each == self.MFroot) or (each == self.BProot) or (each == self.CCroot):  # remove the root GO terms
                continue
            if each not in self.GOParent:
                self.PrintMessage("Warning, the following true GO term is not in GeneOntologyTree, maybe because of version problem?"+str(each))
                continue
            if each not in NewTrue:
                NewTrue[each] = 1

        self.PrintMessage(str(NewPre))
        self.PrintMessage(str(NewTrue))
        # now you get clean pre and true GO sets for evaluation
        return self._CalPrecisionRecall(NewPre,NewTrue)

    # this function is going to propagate your GO term to the root and return a list of unique GO terms (excluding GO root)
    def _propagateGO(self,GOterm):
        self.PrintMessage("Now propagate GO term :"+str(GOterm))
        proGOs = dict()    # this stores all propagated GO terms
        mylist = []    # we use this to do a BFS
        mylist.append(GOterm)
        while True:
            if len(mylist)==0:   # nothing in your list
                break
            temGO = mylist.pop(0)   # pop the first GO term and then add all of his parents
            if temGO in self.GOParent:
                if temGO in proGOs:    # we already add it to the list, no need to propogate
                    continue
                for each in self.GOParent[temGO]:
                    mylist.append(each)
            if (temGO not in proGOs) and (temGO != self.MFroot) and (temGO != self.CCroot) and (temGO != self.BProot):
                proGOs[temGO] = 1
                self.PrintMessage("Adding GO term for propagation " + str(temGO))
        return proGOs

    # this function is going to calculate the precision and recall based on a clean GO sets
    def _CalPrecisionRecall(self,NewPre,NewTrue):
        TP = 0
        FP = 0
        FN = 0
        if len(NewPre)==0 or len(NewTrue)==0:
            #print NewPre, NewTrue
            #sys.exit(0)
            return (-1,-1)     # some of true GO terms are not in the version GO database, so we should not consider these kind of predictions
        for each in NewTrue:
            if each in NewPre:   # great, find a matched one
                TP+=1
            else:
                FN+=1      # you didn't predict, but it's in the true set
        for each in NewPre:
            if each not in NewTrue:
                FP+=1
        precision = float(TP)/(TP+FP)
        recall = float(TP)/(TP+FN)
        return (precision,recall)

    def _displayTree(self):
        print(str(self.GOParent))

# this function is going to load the tree structure, get each GO's parents information
# we only consider is_a information.
    def _loadTree(self,pathOfGOTree):
        preGO = "NULL"
        parents = []
        NameSpace = None
        with open(pathOfGOTree,'r') as fileHandle:
            for lines in fileHandle:
                tem = lines.strip().split()
                if len(tem)<2:
                    continue
                if tem[0] == "id:" and "GO:" in tem[1]:
                    if preGO!="NULL":    # you already have a GO term
                        self.GOParent[preGO] = parents
                        if NameSpace == None:
                            print("Error, why namespace is still none for this ID" + str(preGO)+"=>"+str(lines))
                        else:
                            if preGO not in self.GOSpace:
                                self.GOSpace[preGO] = NameSpace
                            else:
                                if NameSpace != self.GOSpace[preGO]:
                                    print("Conflict, check this ID"+str(preGO)+", it has more than one possible GO trees? "+str(NameSpace)+" and "+str(self.GOSpace[preGO]))

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
            if preGO!="NULL":# add the last GO terms information
                self.GOParent[preGO] = parents
