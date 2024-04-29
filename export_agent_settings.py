#EXAMPLE FOR PULLING OUT THE HIDDEN AGENT ADVANCED SETTINGS
#NEEDS TO DO WORKAROUND TO PULL OUT THE SETTINGS BY CALLING EXPORTAGENTREQUEST

from dfcx_scrapi.core.agents import Agents
from google.cloud.dialogflowcx_v3beta1 import types, services
from google.colab import drive
import zipfile
import io
import os
import json
import pandas as pd

creds_path = ""
agent_id = ""

#Instinate the SCRAPI AGENT CLASS
agent_obj = Agents(creds_path, agent_id)

#Creating the json format, (4) is a json format
json_format = types.agent.ExportAgentRequest.DataFormat(4)

#CREATE THE REQUEST 
request = types.agent.ExportAgentRequest()
request.name = agent_id
request.data_format = json_format

#CREATE THE CLIENT
client = services.agents.AgentsClient(credentials = agent_obj.creds)

#REQUEST EXPORT AGENT BY CLIENT
response = client.export_agent(request)
result = response.result()
content = result.agent_content

#UNZIP THE JSON FILE
with zipfile.ZipFile(io.BytesIO(content)) as z:
    z.extractall()

#PRINT ADVANCEDSETTINGS.SPEECHSETTINGS
with open("agent.json", "r") as f:
    print(json.load(f).get("advancedSettings").get("speechSettings"))


#PULLING OUT THE SPEECHSETTINGS in FLOWS
#@title Settings By Flow
nlu_settings = []
speech_settings = []
advanced_settings = []

for flow in os.listdir("./flows"):
    with open(f"./flows/{flow}/{flow}.json", "r") as f:
        content = json.load(f)
        adv = content.get("advancedSettings")
        if adv:
            advanced_settings.append({"displayName" : flow, **adv})
            spe = adv.get("speechSettings")

        if spe:
            speech_settings.append({"flow" : flow, **spe})

        nlu = content.get("nluSettings")
        if nlu:
            nlu_settings.append({"displayName" : flow, **nlu})

pd.DataFrame(advanced_settings).to_csv("flow_advanced_settings.csv")
pd.DataFrame(nlu_settings).to_csv("flow_nlu_settings.csv")
pd.DataFrame(speech_settings).to_csv("flow_speech_settings.csv")

#PULLING OUT the SPEECH SETTINGS IN PAGES
#@title Settings By Page
nlu_settings = []
speech_settings = []
advanced_settings = []

for flow in os.listdir("./flows"):
    pages = os.listdir(f"./flows/{flow}/pages")
    for page in pages:
        with open(f"./flows/{flow}/{flow}.json", "r") as f:
            content = json.load(f)

        adv = content.get("advancedSettings")
        if adv:
            advanced_settings.append({"displayName" : flow, **adv})
            spe = adv.get("speechSettings")

        if spe:
            speech_settings.append({"flow" : flow, "page" : page, **spe})

        nlu = content.get("nluSettings")
        if nlu:
            nlu_settings.append({"flow" : flow, "page" : page, **nlu})


pd.DataFrame(advanced_settings).to_csv("page_advanced_settings.csv")
pd.DataFrame(nlu_settings).to_csv("page_nlu_settings.csv")
pd.DataFrame(speech_settings).to_csv("page_speech_settings.csv")
