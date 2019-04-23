#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 10:31:56 2019
git clone https://github.com/BretFisher/udemy-docker-mastery.git
@author: tarun.bhavnani@dev.smecorner.com
"""

#to start:

sudo systemctl start docker
#check:
sudo docker run hello-world

sudo docker pull busybox
#The pull command fetches the busybox image from the Docker registry and saves it to our system. 
#You can use the docker images command to see a list of all images on your system.

sudo docker image

#can see both busybox and hello-world

#now run

sudo docker run busybox
#nothing happened!!
#we didn't provide a command, so the container booted up, ran an empty command and then exited.

sudo run busybox echo "hello again"

#to see the running containers
sudo docker ps

#Since no containers are running, we see a blank line. Let's try a more useful variant: 
docker ps -a
#shows all the exited coyainers aswell


#yepp now we want to run more than one command in the container

#we will sh our way in

sudo docker run -it busybox sh

sudo docker ps -a
#remove 
sudo docker rm c01d587ec834 0579280e2007 4c745dc1a62c 4219434c3575

#remove all exited
docker rm $(docker ps -a -q -f status=exited)
#same
sudo docker container prune
#remove images by 
docker rmi


#downloading an image
docker run --rm tbhavnani/catnip

#if you see in my docker tbhavnani/catnip is a repo
#if the image is present locally it takes it, otherwise it goes to the specified docker login 
#and extracts it

#Status: Downloaded newer image for tbhavnani/catnip:latest
 #* Running on http://0.0.0.0:5000/ (Press CTRL+C to quit)

#This site can’t be reached 0.0.0.0 refused to connect.


"we have not specified a port, also we have to run it in detached mode"


docker run -d -P --name cattarun tbhavnani/catnip

-d will run it in detached mode
-P will randomly assign a port

#ceb67fecf33d3ed31d2d20b2d0ca7cbb1f0eb9de841074d85f05d3dac929fee5
#we can see the port where it is launched

sudo docker port cattarun
#5000/tcp -> 0.0.0.0:32768
#can view it at http://0.0.0.0:32768/
#or http://localhost:32768


we can also specify a custom port

sudo docker run -d -p 8888:80 --name cattarun tbhavnani/catnip


#a detached server has to be stopped
docker stop cattarun




"""So basically all we need is to create an image
there is a base image and then a child image, base image is mostly for eg an OS like ubuntu 
or atklest python
some official images can be both base and child
"""

cd docker-curriculum/flask-app

#we will create an image out of this basic flask app


"""Since our application is written in Python, the base image we're going to use will be Python 3. 
"More specifically, we are going to use the python:3-onbuild version of the python image.


Dockerfile
A Dockerfile is a simple text-file that contains a list of commands that the Docker client calls
 while creating an image. It's a simple way to automate the image creation process. 
 The best part is that the commands you write in a Dockerfile are almost identical to their 
 equivalent Linux commands. This means you don't really have to learn new syntax to create 
 your own dockerfiles.


We have to put this Dockerfile in the application directory.
"""
#base image
FROM python:3-onbuild

#The next thing we need to specify is the port number that needs to be exposed. Since our flask app is running on port 5000, 
#that's what we'll indicate.

EXPOSE 5000

#he last step is to write the command for running the application, which is simply - python ./app.py. We use the CMD command to 
#do that -

CMD ["python", "./app.py"]


"""
With that, our Dockerfile is now ready. This is how it looks like -
touch Dockerfile
vi Dockerfile

# our base image
FROM python:3-onbuild

# specify the port number the container should expose
EXPOSE 5000

# run the application
CMD ["python", "./app.py"]
"""

#building image
"The docker build command does the heavy-lifting of creating a Docker image from a Dockerfile"

#lets build this image on tbhavnani, don't forget the preiod

sudo docker build -t tbhavnani/seecat .
#to see if things went well
sudo docker images


sudo docker run -p 8888:5000 tbhavnani/seecat

"""
The command we just ran used port 5000 for the server inside the container, and exposed 
this externally on port 8888. 
Head over to the URL with port 8888, where your app should be live.


 * Running on http://0.0.0.0:5000/ (Press CTRL+C to quit)
the server says 5000 but the app is live on http://0.0.0.0:8888/

"""

#push to the rep


docker push tbhavnani/seecat
#can check on docker web link

#now anybody with docker can play with this
$ docker run -p 8888:5000 tbhavnani/seecat


#create a Dockerrun.aws.json file if to be hosted on aws
#contents of file
"""
{
  "AWSEBDockerrunVersion": "1",
  "Image": {
    "Name": "tbhavnani/seecat",
    "Update": "true"
  },
  "Ports": [
    {
      "ContainerPort": "5000"
    }
  ],
  "Logging": "/var/log/nginx"
}

"""



#########################################################
#########################################################
#Running Ngrok in a container using docker
https://technology.amis.nl/2019/01/06/expose-docker-container-services-on-the-internet-using-ngrok/
https://github.com/gtriggiano/ngrok-tunnel

#########################################################
#########################################################




###################3


sudo docker network ls
sudo docker network create tarun

sudo docker container run -d -p 80:80 --network tarun --name nginx nginx
sudo docker container run -d -p 3306:3306 --network tarun --name mysql -e MYSQL_RANDOM_ROOT_PASSWORD=true mysql
sudo docker container run -d -p 8080:80 --network tarun --name httpd httpd

sudo docker network inspect tarun

-d is for --detached
-p is for port
--name is name
-e is environment
--network is for network

############################################3

build images
-f is for Dockerfile
































