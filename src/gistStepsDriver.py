import sys
import os
import json
from gistSteps import step1, step2, step3, step4, step5, step6, step7, step8, step9

selectedStep = sys.argv[1]
selectedInputFile = sys.argv[2]

def createNewUserSession():
    newUserSession = {
        'sessionValues': {},
        'gistInstance': {}
    }
    return newUserSession

with open(selectedInputFile, 'r') as file:
    inputData = json.load(file)

session_path = "exampleSession.json"

if os.path.exists(session_path):
    with open(session_path, 'r') as file:
        currentSession = json.load(file)
else:
    currentSession = createNewUserSession()

switch = {
    'step1': step1,
    'step2': step2,
    'step3': step3,
    'step4': step4,
    'step5': step5,
    'step6': step6,
    'step7': step7,
    'step8': step8,
    'step9': step9
}

def default_case(input):
    return "No step found"

curStep = switch.get(selectedStep, default_case)
output = curStep(inputData, currentSession)

with open(session_path, 'w') as file:
    json.dump(output, file, indent=4)

