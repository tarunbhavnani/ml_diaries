
#give permissions to sh file
sudo chmod +x ./scripts/start_services.sh 

sudo docker build -t rasa-chatbot-d .

sudo docker run -it --rm -p 5005:5005 -e PORT=5005 -e SLACK="xoxb-542065604356-582580159955-nuwuIUEQfJQuYGbOWf3MDpq" rasa-chatbot-d


```
It starts a webserver with rest api and listens for messages at localhost:5005

#### Test over REST api

```bash
curl --request POST \
  --url http://localhost:5005/webhooks/rest/webhook \
  --header 'content-type: application/json' \
  --data '{
    "message": "Hi"
  }'
```
**Response**
```http
HTTP/1.1 200 OK
Content-Type: application/json
Content-Length: 59
Access-Control-Allow-Origin: *

[{
  "recipient_id": "default",
  "text": "Hi, how is it going?"
}]
```

#### Run using docker compose
Optionally to run the actions server in separate container start the services using docker-compose. The action server runs on http://action_server:5055/webhook (docker's internal network). The rasa-core services uses the config/endpoints.local.yml to find this actions server

```
docker-compose up
```
#### Train Dialog model
The repository already contains a trained dialog model at models/rasa_core. To retrain the model you can run:
```powershell
docker run --rm -v ${PWD}:/app rasa-chatbot python3 \
           -m rasa_core.train -d /app/domain.yml \
           -s /app/data/stories.md \
           -o /app/models/rasa_core
```
#### Train NLU model
The repository already contains the trained NLU model at models/rasa_nlu. To retrain NLU model you can run:

```powershell
docker run --rm -v ${PWD}:/app rasa-chatbot python3 \
           -m rasa_nlu.train -c /app/config/nlu_config.yml \
           -d /app/data/nlu.md --fixed_model_name nlu \
           -o /app/models/rasa_nlu --project current
```

## Deploy to Heroku
On heroku free tier we can start two containers using two dynos, but there isn't a way for the containers to communicate with each other on Heroku. So, we push everything (actions server/rasa core/nlu) in the same container.

```bash
heroku container:push web
heroku container:release web
```

Another option would be to create a separate app altogether for actions server (nlu server can also be run as a separate app), which then can communicate with each other over http.

## Integration with Facebook
rasa supports integration with multiple channels. Apart from exposing the REST api over http, we can integrate with facebook. 

Go to https://developers.facebook.com and creat an app. We can handle messages sent to a facebook page from our app. To do so add messenger to the facebook app and subscribe to a page. Update app secret and page tocken in config/credentials.yml. On the facebook app, update the webhook url to the deployed heroku app (https://rasa-chatbot.herokuapp.com/webhooks/facebook/webhook).


