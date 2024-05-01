import re
import os
import time
from collections import defaultdict
from typing import Dict, Any

import numpy as np
import pandas as pd

from dfcx_scrapi.builders.response_messages import ResponseMessageBuilder
from dfcx_scrapi.core.agents import Agents
from dfcx_scrapi.core.flows import Flows
from dfcx_scrapi.core.intents import Intents
from dfcx_scrapi.core.pages import Pages
from dfcx_scrapi.core.transition_route_groups import TransitionRouteGroups
from dfcx_scrapi.tools.dataframe_functions import DataframeFunctions

from google.cloud.dialogflowcx_v3beta1 import types

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = creds_path

# Instantiate the core classes
agents_instance = Agents(creds_path=creds_path)
flows_instance = Flows(creds_path=creds_path)
intents_instance = Intents(creds_path=creds_path)
pages_instance = Pages(creds_path=creds_path)
route_groups_instance = TransitionRouteGroups(creds_path=creds_path)

# Instantiate the tools classes
dffx_instance = DataframeFunctions(creds_path=creds_path)



# Get the Agent and its ID
my_agent = agents_instance.get_agent_by_display_name(
    project_id=project_id, display_name=agent_display_name
)
agent_id = my_agent.name

# Load the flows
all_flows = flows_instance.list_flows(agent_id=agent_id)
flows_map_reverse = flows_instance.get_flows_map(agent_id=agent_id, reverse=True)

intents_map = intents_instance.get_intents_map(agent_id=agent_id)
time.sleep(3)


all_pages_per_flow, all_route_groups_per_flow = {}, {}

for flow_name, flow_id in flows_map_reverse.items():
    # Load the pages
    all_pages_per_flow[flow_name] = pages_instance.list_pages(flow_id=flow_id)
    time.sleep(1)

    # Load the transition route groups
    route_groups_list = route_groups_instance.list_transition_route_groups(
        flow_id=flow_id
    )
    all_route_groups_per_flow[flow_name] = route_groups_list
    time.sleep(1)


class AllPagesCustomDict(dict):

    def __missing__(self, key):
        if (isinstance(key, str) and key.endswith("PAGE")):
            return str(key).rsplit("/", maxsplit=1)[-1]
        elif (isinstance(key, str) and key.endswith("END_SESSION")):
            return "END_SESION"
        else:
            return np.nan


all_pages_map = AllPagesCustomDict()
for flow_name, page_list in all_pages_per_flow.items():
    for page in page_list:
        all_pages_map[page.name] = page.display_name

all_pages_map.update({f.name: f.display_name for f in all_flows})

def page_link_creator(page_id: str) -> str:
    """Returns a link to the DFCX's page which has page_id."""
    base_endpoint = "https://dialogflow.cloud.google.com/cx"
    project_agent_flow_components = "/".join(page_id.split("/")[:-2])
    page_id_query = f"flow_creation?pageId={page_id.split('/')[-1]}"

    link = f"{base_endpoint}/{project_agent_flow_components}/{page_id_query}"
    return link

def start_page_link_creator(flow_id: str) -> str:
    """Returns a link to the DFCX Start Page of a flow with flow_id."""
    base_endpoint = "https://dialogflow.cloud.google.com/cx"
    # project_agent_flow_components = "/".join(flow_id.split("/")[:-2])
    page_id_query = "flow_creation?pageId=START_PAGE"

    link = f"{base_endpoint}/{flow_id}/{page_id_query}"
    return link

def route_group_link_creator(route_group_id: str) -> str:
    """Returns a link to the DFCX's TransitionRouteGroup with route_group_id."""
    base_endpoint = "https://dialogflow.cloud.google.com/cx"
    project_agent_flow_components = "/".join(route_group_id.split("/")[:6])
    trg_id_query = f"transitionRouteGroups?flowId={route_group_id.split('/')[-3]}&routeGroupId={route_group_id.split('/')[-1]}"

    link = f"{base_endpoint}/{project_agent_flow_components}/{trg_id_query}"
    return link

def location_finder(
    flow: str, page: str, trg: str,
    intent: str, cond: str, event: str, param: str,
    target_type: str, target_id: str
) -> Dict[str, str]:
    """Returns a dictionary with provided argumentsand set empty strings to None."""
    out = {
        "Flow": flow, "Page": page, "TransitionRouteGroup": trg,
        "Intent": intent, "Condition": cond, "Event": event, "Param": param,
        "TargetType": target_type, "TargetId": target_id,
    }

    for k, v in out.items():
        if v == "":
            out[k] = None

    return out

def target_type_finder(tr) -> str:
    """Return a target type of some route(TransitionRoute/EventHandler)."""
    if tr.target_flow:
        return "Flow"
    elif tr.target_page:
        return "Page"
    else:
        return None

def target_id_finder(tr) -> str:
    """Return a target id of some route(TransitionRoute/EventHandler)."""
    if tr.target_flow:
        return tr.target_flow
    elif tr.target_page:
        return tr.target_page
    else:
        return None


def fulfillment_generator() -> Dict[str, Any]:
    """Main function for the fulfillment generator.
    It loops through all of the flows (START_PAGEs), all the pages of each flow,
      and all the transition routes of each flow. For each fulfillment it faces,
      it will return a dictionary with the following content:
        "fulfillment": The fulfillment proto obj.
        "location": The location of the fulfillment as a dictionary.
        "parent_proto": The parent proto obj of the fulfillment.
          In order to update the fulfillment after making changes,
          we need to update the parent proto.
        "link": Link to the parent proto obj in DFCX UI for ease of use.
    """
    # Flows
    for flow in all_flows:
        # Cerate link
        flow_link = start_page_link_creator(flow.name)
        # Routes
        for tr in flow.transition_routes:
            yield {
                "fulfillment": tr.trigger_fulfillment,
                "location": location_finder(
                    flow=flow.display_name, page="START_PAGE", trg=None,
                    intent=intents_map.get(tr.intent), cond=tr.condition,
                    event=None, param=None,
                    target_type=target_type_finder(tr), target_id=target_id_finder(tr)
                ),
                "parent_proto": flow,
                "link": flow_link,
            }

        for eh in flow.event_handlers:
            yield {
                "fulfillment": eh.trigger_fulfillment,
                "location": location_finder(
                    flow=flow.display_name, page="START_PAGE", trg=None,
                    intent=None, cond=None, event=eh.event, param=None,
                    target_type=target_type_finder(eh), target_id=target_id_finder(eh)
                ),
                "parent_proto": flow,
                "link": flow_link,
            }

    # Pages
    for flow_name in all_pages_per_flow:
        for page in all_pages_per_flow[flow_name]:
            # Create Page link
            page_link = page_link_creator(page.name)
            # Entry Fulfillment
            yield {
                "fulfillment": page.entry_fulfillment,
                "location": location_finder(
                    flow=flow_name, page=page.display_name, trg=None,
                    intent=None, cond=None, event=None, param=None,
                    target_type=None, target_id=None
                ),
                "parent_proto": page,
                "link": page_link,
            }

            # Parameters
            for param in page.form.parameters:
                yield {
                    "fulfillment": param.fill_behavior.initial_prompt_fulfillment,
                    "location": location_finder(
                        flow=flow_name, page=page.display_name, trg=None,
                        intent=None, cond=None, event=None, param=param.display_name,
                        target_type=None, target_id=None
                    ),
                    "parent_proto": page,
                    "link": page_link,
                }

                for eh in param.fill_behavior.reprompt_event_handlers:
                    yield {
                        "fulfillment": eh.trigger_fulfillment,
                        "location": location_finder(
                            flow=flow_name, page=page.display_name, trg=None,
                            intent=None, cond=None, event=eh.event, param=param.display_name,
                            target_type=target_type_finder(eh), target_id=target_id_finder(eh)
                        ),
                        "parent_proto": page,
                        "link": page_link,
                    }

            # Routes
            for tr in page.transition_routes:
                yield {
                    "fulfillment": tr.trigger_fulfillment,
                    "location": location_finder(
                        flow=flow_name, page=page.display_name, trg=None,
                        intent=intents_map.get(tr.intent), cond=tr.condition, event=None, param=None,
                        target_type=target_type_finder(tr), target_id=target_id_finder(tr)
                    ),
                    "parent_proto": page,
                    "link": page_link,
                }
            for eh in page.event_handlers:
                yield {
                    "fulfillment": eh.trigger_fulfillment,
                    "location": location_finder(
                        flow=flow_name, page=page.display_name, trg=None,
                        intent=None, cond=None, event=eh.event, param=None,
                        target_type=target_type_finder(eh), target_id=target_id_finder(eh)
                    ),
                    "parent_proto": page,
                    "link": page_link,
                }

    # TransitionRouteGroups
    for flow_name in all_route_groups_per_flow:
        for trg in all_route_groups_per_flow[flow_name]:
            # Create Route Group link
            trg_link = route_group_link_creator(trg.name)
            for tr in trg.transition_routes:
                yield {
                    "fulfillment": tr.trigger_fulfillment,
                    "location": location_finder(
                        flow=flow_name, page=None, trg=trg.display_name,
                        intent=intents_map.get(tr.intent), cond=tr.condition, event=None, param=None,
                        target_type=target_type_finder(tr), target_id=target_id_finder(tr)
                    ),
                    "parent_proto": trg,
                    "link": trg_link,
                }

target_unicodes_records = []
overall_unicodes_records = []

ff_gen = fulfillment_generator()

while True:
    try:
        next_ff = next(ff_gen)
    except StopIteration:
        break

    # Get the fulfillment parts
    the_fulfillment = next_ff["fulfillment"]
    location = next_ff["location"]
    parent_proto = next_ff["parent_proto"]
    link = next_ff["link"]

    # New Code goes here
    for msg in the_fulfillment.messages:
        if msg.text:
            for txt in msg.text.text:
                for ch in txt:
                    if ord(ch) in target_unicodes:
                        target_unicodes_records.append({
                            **location, **{"Link": link},
                            **{"Fulfillment": the_fulfillment.messages},
                            **{"utf_code": f"U+{ord(ch):04X}"}
                        })
                        
                    overall_unicodes_records.append({ 
                        ** location, **{"Link": link},
                        **{"Fulfillment": the_fulfillment.messages},
                        **{"utf_code": f"U+{ord(ch):04X}"}
                    })

overall_unicodes_df = pd.DataFrame.from_records(overall_unicodes_records)
target_unicodes_df = pd.DataFrame.from_records(target_unicodes_records)

# Write to sheets
dffx_instance.dataframe_to_sheets(sheet_name, worksheet_name, target_unicodes_df)
