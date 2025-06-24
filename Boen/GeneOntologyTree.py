"""
Author: Dr. Renzhi Cao
Written: 1/1/2020
Any questions, please email: caora@plu.edu
"""
import sys

class GeneOntologyTree:
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

    def GOSimilarity(self, GO1, GO2):
        self.PrintMessage("Now compute the similarity of two GO terms sets based on GO tree. We may need to DFS to get the path from GO to the root.")
        if GO1 not in self.GOSpace or GO2 not in self.GOSpace:
            return -1

        if self.GOSpace[GO1] != self.GOSpace[GO2]:
            return 0

        Longest2Share = 0
        CommonStep = 0
        Level1, Sets1 = [GO1], {GO1: 1}
        Level2, Sets2 = [GO2], {GO2: 1}
        StartingLevel = None

        while True:
            tag = 0
            for eachGO in Level1:
                if eachGO in Sets2:
                    StartingLevel = Level1
                    tag = 1
            if tag == 1: break
            for eachGO in Level2:
                if eachGO in Sets1:
                    StartingLevel = Level2
                    tag = 1
            if tag == 1: break
            Longest2Share+=1
            
            temLevel = []
            for eachGO in Level1:
                if (eachGO != self.MFroot) and (eachGO != self.BProot) and (eachGO != self.CCroot) and (eachGO in self.GOParent):
                    for tGO in self.GOParent[eachGO]:
                        temLevel.append(tGO)
                        Sets1[tGO] = 1
            Level1 = temLevel

            temLevel = []
            for eachGO in Level2:
                if (eachGO != self.MFroot) and (eachGO != self.BProot) and (eachGO != self.CCroot) and (eachGO in self.GOParent):
                    for tGO in self.GOParent[eachGO]:
                        temLevel.append(tGO)
                        Sets2[tGO] = 1
            Level2 = temLevel

            if not Level1 or not Level2:
                return 1.0 / (Longest2Share + 1.0)

        while True:
            CommonStep+=1
            temLevel = []
            for eachGO in StartingLevel:
                if (eachGO != self.MFroot) and (eachGO != self.BProot) and (eachGO != self.CCroot) and (eachGO in self.GOParent):
                    for tGO in self.GOParent[eachGO]:
                        temLevel.append(tGO)
            StartingLevel = temLevel
            if not StartingLevel: break

        return float(CommonStep) / (CommonStep + Longest2Share)

    def MaxSimilarity(self, PredictionSet = [], TrueSet = []):
        maxSim = 0.0
        for preGO in PredictionSet:
            for tGO in TrueSet:
                temSim = self.GOSimilarity(preGO,tGO)
                if temSim > maxSim:
                    maxSim = temSim
        return maxSim
        
    def GetGONameSpace(self, GO):
        return self.GOSpace.get(GO)

    def GOSetsPropagate(self, PredictionSet, TrueSet):
        if not PredictionSet or not TrueSet:
            return (0,0)
        
        NewPre = {}
        for each in PredictionSet:
            if "GO:" not in each: each = "GO:"+each
            if (each == self.MFroot) or (each == self.BProot) or (each == self.CCroot): continue
            if each not in self.GOParent: continue
            for newGO in self._propagateGO(each):
                if newGO not in NewPre: NewPre[newGO] = 1
        
        NewTrue= {}
        for each in TrueSet:
            if "GO:" not in each: each = "GO:"+each
            if (each == self.MFroot) or (each == self.BProot) or (each == self.CCroot): continue
            if each not in self.GOParent: continue
            for newGO in self._propagateGO(each):
                if newGO not in NewTrue: NewTrue[newGO] = 1

        return self._CalPrecisionRecall(NewPre,NewTrue)

    def _propagateGO(self,GOterm):
        proGOs, mylist = {}, [GOterm]
        while mylist:
            temGO = mylist.pop(0)
            if temGO in self.GOParent:
                if temGO in proGOs: continue
                for each in self.GOParent[temGO]:
                    mylist.append(each)
            if (temGO not in proGOs) and (temGO != self.MFroot) and (temGO != self.CCroot) and (temGO != self.BProot):
                proGOs[temGO] = 1
        return proGOs

    def _CalPrecisionRecall(self,NewPre,NewTrue):
        if not NewPre or not NewTrue:
            return (0.0, 0.0) if not NewTrue else (1.0 if not NewPre else 0.0, 0.0)
            
        TP = float(len(set(NewPre.keys()) & set(NewTrue.keys())))
        FP = float(len(set(NewPre.keys()) - set(NewTrue.keys())))
        FN = float(len(set(NewTrue.keys()) - set(NewPre.keys())))
        
        precision = TP/(TP+FP) if (TP+FP) > 0 else 0.0
        recall = TP/(TP+FN) if (TP+FN) > 0 else 0.0
        return (precision,recall)

    def _loadTree(self,pathOfGOTree):
        preGO, parents, NameSpace = "NULL", [], None
        try:
            with open(pathOfGOTree,'r') as fileHandle:
                for lines in fileHandle:
                    tem = lines.strip().split()
                    if len(tem)<2: continue
                    if tem[0] == "id:" and "GO:" in tem[1]:
                        if preGO!="NULL":
                            self.GOParent[preGO] = parents
                            if NameSpace:
                                self.GOSpace[preGO] = NameSpace
                        preGO = tem[1]
                        parents, NameSpace = [], None
                    elif tem[0] == "is_a:":
                        parents.append(tem[1])
                    elif tem[0] == "is_obsolete:" and tem[1] == "true":
                        preGO, parents, NameSpace = "NULL", [], None
                    elif tem[0] == "namespace:":
                        NameSpace = tem[1]
                if preGO!="NULL":
                    self.GOParent[preGO] = parents
                    if NameSpace: self.GOSpace[preGO] = NameSpace
        except FileNotFoundError:
            print(f"FATAL ERROR: The Gene Ontology OBO file was not found at '{pathOfGOTree}'")
            print("Please check the path and try again.")
            sys.exit(1)