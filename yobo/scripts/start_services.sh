### This script is triggered from within docker contrainer
### to start multiple processes in the same container.
### This script is defined in the CMD option in Dockerfile

# Start actions server in background
python3 -m rasa_core_sdk.endpoint --actions app.actions.actions&

# Start rasa core server with nlu model
#python3 -m rasa_core.run --enable_api --core /app/models/rasa_core -u /app/models/rasa_nlu/current/nlu --endpoints /app/config/endpoints.yml --credentials /app/config/credentials.yml -p $PORT



#fir up ngrok

#./ngrok http 5005

# Start rn_app.py
python3 /app/scripts/run_app_d.py $SLACK
#python3 /app/scripts/run_app_d.py "xoxb-542065604356-582580159955-nuwuIUEQfJQ"

