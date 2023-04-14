# -*- coding: utf-8 -*-
"""
Created on Tue Oct 12 21:49:42 2021

@author: ELECTROBOT
"""

def convert_park_to_dict(park):
    park_dict = {}
    for attr in vars(park):
        print(attr)
        attr_value = getattr(park, attr)
        if isinstance(attr_value, Point):
            point_dict = vars(attr_value)
            park_dict[attr] = point_dict
        else:
            park_dict[attr] = attr_value
    return park_dict


def test():
    park = Park(1, 'my_house', ['swings', 'shop'], 50, 60)
    park_dict = convert_park_to_dict(park)
    return jsonify(park_dict)

class Park(object):
    park_id = 0
    address = ""
    services = []
    position = None

    def __init__(self, park_id, address, services, latitude, longitude):
        self.park_id = park_id
        self.address = address
        self.services = services
        self.position = Point(latitude, longitude)
class Point(object):
    latitude = None
    longitude = None

    def __init__(self, latitude, longitude):
        self.latitude = latitude
        self.longitude = longitude

# =============================================================================
# 
# =============================================================================
import json

data = {
    'president': {
        "name": """Mr. Presidente""",
        "male": True,
        'age': 60,
        'wife': None,
        'cars': ('BMW', "Audi")
    }
}

# serialize
json_data = json.dumps(data, indent=2)

print(json_data)
# {
#   "president": {
#     "name": "Mr. Presidente",
#     "male": true,
#     "age": 60,
#     "wife": null,
#     "cars": [
#       "BMW",
#       "Audi"
#     ]
#   }
# }

# deserialize
restored_data = json.loads(json_data) # deserialize