FROM ubuntu:18.04
MAINTAINER tbhavnani

RUN apt-get update && apt-get install -y python3 python3-pip
RUN pip3 install rasa-core rasa-nlu[tensorflow] spacy sklearn-crfsuite rasa-core-sdk nltk word2number argparse
RUN python3 -m spacy download en

ADD ./models /app/models/
ADD ./config /app/config/
ADD ./actions /app/actions/
ADD ./scripts /app/scripts/


ENTRYPOINT []
CMD /app/scripts/start_services.sh
#CMD python3 -m rasa_core_sdk.endpoint --actions app.actions.actions& && python3 -m rasa_core.run --enable_api --core /app/models/rasa_core -u /app/models/rasa_nlu/current/nlu --endpoints /app/config/endpoints.yml --credentials /app/config/credentials.yml -p $PORT
# python3 -m rasa_nlu.train -c config/nlu_config.yml -d data/nlu.md --fixed_model_name nlu -o models/rasa_nlu --project current
