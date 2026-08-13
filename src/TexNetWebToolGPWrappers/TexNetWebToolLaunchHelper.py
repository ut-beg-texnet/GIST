from ast import Dict
from email import message
import json
import mimetypes

import os
import shutil

import pandas as pd

from .Constants import *
from .Message import Message

class TexNetWebToolLaunchHelper(object):
    """description of class"""
    
    _scratchPath = None
    _finalArgsFilePath = None
    _origArgsData = None
    _resultsFilePath = None
    _argsData = None
    _success = True
    _messages = []

    @property
    def scratchPath(self):
        return self._scratchPath
    
    @property
    def finalArgsFilePath(self):
        return self._finalArgsFilePath
    
    @property
    def origArgsData(self):
        return self._origArgsData
    
    @property
    def argsData(self):
        return self._argsData
    
    @argsData.setter
    def argsData(self, newVal):
        self._argsData = newVal
    
    def __init__(self, scratchPath):

        self._scratchPath = scratchPath
        
        self.loadOrigArgsData()
        
        
    def loadOrigArgsData(self):
        argsFilePath = os.path.join(self._scratchPath, ARGS_FILE_NAME)
        
        jsonContents = ""

        with open(argsFilePath, "r") as fh:
            jsonContents = fh.read()
    
        self._origArgsData = json.loads(jsonContents)    
    
    def getParameterStateWithStepIndexAndParamName(self, stepIndex, paramName):
        """Returns the parameter state object (as a dictionary) with the given name from the step at the given index"""
        ret = None

        if self._origArgsData != None and len(self._origArgsData["SessionState"]["StepState"]) > stepIndex:
            try:
                step = self._origArgsData["SessionState"]["StepState"][stepIndex]
                
                for param in step["InputParameterStates"]:
                    if param["ProcessStepParameterName"] == paramName:
                        ret = param
                        break

                if ret == None:
                    for param2 in step["OutputParameterStates"]:
                        if param2["ProcessStepParameterName"] == paramName:
                            ret = param2
                            break
            except:
                pass

        return ret

    def getParameterValueWithStepIndexAndParamName(self, stepIndex, paramName):
        """Returns the value of the parameter with the given name from the step at the given index"""
        
        ret = None
        
        param = self.getParameterStateWithStepIndexAndParamName(stepIndex, paramName)
        
        if param != None and param["Value"] != None:
            
            #For each 
            if param["DataType"] == PARAMETER_TYPE_INT:
                ret = int(param["Value"])
            elif param["DataType"] == PARAMETER_TYPE_FLOAT:
                ret = float(param["Value"])
            elif param["DataType"] == PARAMETER_TYPE_BOOLEAN:
                ret = bool(param["Value"])
            else:
                ret = param["Value"]

        return ret

    def getDatasetFilePathWithStepIndexAndParamName(self, stepIndex, paramName):
        
        ret = None
        
        paramVal = self.getParameterValueWithStepIndexAndParamName(stepIndex, paramName)
        
        if paramVal != None:
            ret = self._origArgsData["DatasetPaths"][str(paramVal)]
 
        return ret
        
    def getDatasetFileContentsWithStepIndexAndParamNameAsDataFrame(self, stepIndex, paramName):
        
        dfRet = None

        path = self.getDatasetFilePathWithStepIndexAndParamName(stepIndex, paramName)

        if path != None:
            dfRet = pd.read_csv(path)

        return dfRet
     
    def setParamValueWithStepIndexAndParamName(self, stepIndex, paramName, value):
        """Sets the value of the parameter with the given name from the step at the given index"""
        
        param = self.getParameterStateWithStepIndexAndParamName(stepIndex, paramName)
        
        if param != None:
            if param["DataType"] == PARAMETER_TYPE_USER_DATASET_ROW:
                param["Value"] = value
            else:
                param["Value"] = str(value)
        else:
            raise Exception("Parameter was not found.")
        
    def saveDataFrameAsParameterWithStepIndexAndParamName(self, stepIndex, paramName, df):
        
        #First, we need to figure out the filename.
        path = ""
        tries = 0
        
        while path == "" or os.path.exists(path) == True:
            
            filename = ""
            
            if tries == 0:
                filename = paramName + ".csv"
            else:
                filename = paramName + "_" + str(tries) + ".csv"

            path = os.path.join(self._scratchPath, filename)
            
            tries = tries + 1
            
        df.to_csv(path, index=False)

        self.setParamValueWithStepIndexAndParamName(stepIndex, paramName, path)

    def saveFileAsParameterWithStepIndexAndParamName(self, stepIndex, paramName, srcPath):
        """Registers an existing file on disk as a parameter, without loading it into a DataFrame.

        Same filename collision-avoidance and parameter registration as
        saveDataFrameAsParameterWithStepIndexAndParamName, but for callers that already have the
        desired CSV on disk verbatim (e.g. passing an input dataset through unmodified) and would
        otherwise pay for a wasted pd.read_csv/df.to_csv round-trip on a large file.
        """

        #First, we need to figure out the filename.
        path = ""
        tries = 0

        while path == "" or os.path.exists(path) == True:

            filename = ""

            if tries == 0:
                filename = paramName + ".csv"
            else:
                filename = paramName + "_" + str(tries) + ".csv"

            path = os.path.join(self._scratchPath, filename)

            tries = tries + 1

        shutil.copyfile(srcPath, path)

        self.setParamValueWithStepIndexAndParamName(stepIndex, paramName, path)

    def saveGraphArtifact(self, key, title, renderer, path, contentType=None, caption=None, displayOrder=0, preferredHeight=None):
        """Registers a tool-generated graph artifact for portal rendering."""
        if path is None or str(path).strip() == "":
            raise ValueError("Graph artifact path is required.")

        fullPath = os.path.abspath(path)
        scratchRoot = os.path.abspath(self._scratchPath)

        if not fullPath.startswith(scratchRoot + os.sep):
            raise ValueError("Graph artifact path must be inside the scratch directory.")

        if not os.path.exists(fullPath):
            raise FileNotFoundError("Graph artifact file was not found: " + fullPath)

        relativePath = os.path.relpath(fullPath, scratchRoot)

        if contentType is None:
            contentType = mimetypes.guess_type(fullPath)[0] or "application/octet-stream"

        artifact = {
            "key": key,
            "title": title,
            "caption": caption,
            "renderer": renderer,
            "path": relativePath,
            "contentType": contentType,
            "displayOrder": displayOrder,
            "preferredHeight": preferredHeight
        }

        if "GraphArtifacts" not in self._origArgsData or self._origArgsData["GraphArtifacts"] is None:
            self._origArgsData["GraphArtifacts"] = []

        self._origArgsData["GraphArtifacts"].append(artifact)
    
    def getScratchDataPathValue(self, paramValue):
        
        return self._scratchPath

    def writeFinalArgsFile(self, outputPath):
        sFinalArgs = json.dumps(self.argsData)
        
        with open(outputPath, "w") as fh:
            fh.write(sFinalArgs)
            

    def writeResultsFile(self):
        
        origArgsData = self._origArgsData

        sResultsData = json.dumps(origArgsData, sort_keys=True, indent=4)
        
        outputPath = os.path.join(self.scratchPath, "results.json")

        with open(outputPath, "w") as fh:
            fh.write(sResultsData)
            
    def setSuccessForStepIndex(self, stepIndex, newSuccessState):
        step = self._origArgsData["SessionState"]["StepState"][stepIndex]
        
        step["Success"] = newSuccessState
            
    def addMessageWithStepIndex(self, stepIndex, messageContent, messageLevel):
        
        if self._origArgsData != None and len(self._origArgsData["SessionState"]["StepState"]) > stepIndex:
            try:
                step = self._origArgsData["SessionState"]["StepState"][stepIndex]

                msg = {}
                msg["MessageContent"] = messageContent
                msg["MessageLevel"] = messageLevel
                
                step["Messages"].append(msg)
            except ex:
                pass
        
            
        
