ENVIRONMENT='LOCAL'

def __read_file__():
    global ENVIRONMENT
    try:
        with open('/config/environment','r') as f:
            ENVIRONMENT=f.read()
            ENVIRONMENT=ENVIRONMENT.upper()
    except:
        pass

__read_file__()

