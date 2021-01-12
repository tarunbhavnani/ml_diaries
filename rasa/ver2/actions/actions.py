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