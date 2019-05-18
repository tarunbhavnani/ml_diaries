#os.chdir('/home/tarun.bhavnani@dev.smecorner.com/Desktop/final_bot/final_bot2')
from rasa_core.channels.slack import SlackInput
from rasa_core.agent import Agent
from rasa_core.interpreter import RasaNLUInterpreter
import yaml
from rasa_core.utils import EndpointConfig
from rasa_core.tracker_store import MongoTrackerStore


nlu_interpreter = RasaNLUInterpreter('/app/models/rasa_nlu/current/nlu')

core='/app/models/rasa_core'

action_endpoint = EndpointConfig(url="http://localhost:5055/webhook")



agent = Agent.load(path=core, interpreter = nlu_interpreter, action_endpoint = action_endpoint)


#input_channel = SlackInput('xoxb-542065604356-542500977682-faR2rS0xAcTANpn4wAU8hAiF') #your bot user authentication token
input_channel = SlackInput('xoxb-542065604356-582580159955-lwMnFo4PGvUc4DopgvWrpkrL') #your bot user authentication token
agent.handle_channels([input_channel], 5005, serve_forever=True)

