from typing import Any, Text, Dict, List
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
#from rasa_core_sdk import Action
from rasa_core_sdk.events import SlotSet


class Actionemail(Action):
	
    def name(self,) -> Text:
    	return 'action_email'

    #def run(self, dispatcher, CollectingDispatcher,tracker: Tracker,domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
    def run(self, dispatcher, tracker, domain):
    	dispatcher.utter_message("hello from tarun")
    	last_message= tracker.latest_message['text']
    	sender_id= tracker.sender_id
    	all_events=tracker.events
    	
    	
    	#dispatcher.utter_message(last_message)
    	#dispatcher.utter_message(sender_id)
    	for item in all_events:
    		for dic in item:
    			dispatcher.utter_message(str(item[dic]))
    	



    	return []
class ActionDefaultAskAffirmation(Action):
   """Asks for an affirmation of the intent if NLU threshold is not met."""

   def name(self):
       return "action_default_ask_affirmation"

   def __init__(self):
       self.intent_mappings = {'greet':'greet','bot_name':'bot_name','abuse':'abuse','chitchat':'chitchat','goodbye':'goodbye'}
       # read the mapping from a csv and store it in a dictionary
       #with open('intent_mapping.csv', newline='', encoding='utf-8') as file:
       #    csv_reader = csv.reader(file)
       #    for row in csv_reader:
       #        self.intent_mappings[row[0]] = row[1]

   def run(self, dispatcher, tracker, domain):
       # get the most likely intent
       last_intent_name = tracker.latest_message['intent_ranking'][1]['name']
       

       # get the prompt for the intent
       intent_prompt = self.intent_mappings[last_intent_name]

       # Create the affirmation message and add two buttons to it.
       # Use '/<intent_name>' as payload to directly trigger '<intent_name>'
       # when the button is clicked.
       message = "Did you mean '{}'?".format(intent_prompt)
       buttons = [{'title': 'Yes',
                   'payload': '/{}'.format(last_intent_name)},
                  {'title': 'No',
                   'payload': '/out_of_scope'}]
       dispatcher.utter_button_message(message, buttons=buttons)

       return []
#class ActionCheckRestaurants(Action):
#   def name(self):
      # type: () -> Text
#      return "action_check_restaurants"


#   def run(self, dispatcher, tracker, domain):
      # type: (CollectingDispatcher, Tracker, Dict[Text, Any]) -> List[Dict[Text, Any]]

#      cuisine = tracker.get_slot('cuisine')
#      q = "select * from restaurants where cuisine='{0}' limit 1".format(cuisine)
#      result = db.query(q)

#      return [SlotSet("matches", result if result is not None else [])]