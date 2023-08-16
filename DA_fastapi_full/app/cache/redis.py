# -*- coding: utf-8 -*-
"""
Created on Wed Aug 16 11:03:35 2023

@author: tarun
"""

import redis
from .cache import CacheInterface
from ..config import CLIENT_CERT, CLIENT_KEY, REDIS_HOST, CERT_CA, REDIS_HOST_SLAVE

import os
import random

class RedisCache(CacheInterface):
    def __init__(self):
        self.redis_client_server= redis.Redis(host= REDIS_HOST, port=443, ssl=True,
                                              password= os.environ.get("REDIS_PASSWORD"),
                                              ssl_certfile=CLIENT_CERT,
                                              ssl_cert_reqs= "required",
                                              ssl_ca_certs=CERT_CA)
        self.redis_client_read= redis.Redis(host= REDIS_HOST_SLAVE, port=443, ssl=True,
                                              password= os.environ.get("REDIS_PASSWORD"),
                                              ssl_certfile=CLIENT_CERT,
                                              ssl_cert_reqs= "required",
                                              ssl_ca_certs=CERT_CA)
        
    def __get_random_server(self):
        """
        

        Returns
        random server bet master and slave
        None.

        """
        return self.redis_client_read
    
    def ping(self):
        """
        pings master and read

        Returns
        -------
        None.

        """
        server_ping= False
        read_ping=False
        try:
            server_ping= self.redis_client_server.ping()
        except:
            pass
        
        try:
            read_ping= self.redis_client_server.ping()
        except:
            pass
        return {"master":server_ping, "read":read_ping}
    
    def info(self, server=None):
        """
        returns redis master slave info

        Parameters
        ----------
        server : TYPE, optional
            DESCRIPTION. The default is None.

        Returns
        -------
        None.

        """
        if not server:
            return {"master": self.redis_client_server.info(), "read": self.redis_client_read.info()}
        
        elif server=="master":
            return {"master": self.redis_client_server.info()}
        elif server=="read":
            return {"read": self.redis_client_read.info()}
        return {"master": self.redis_client_server.info(), "read": self.redis_client_read.info()}
    
    def write_to_cache_user(self, user_id, obj):
        """
        saves key value to server

        Parameters
        ----------
        user_id : TYPE
            DESCRIPTION.
        obj : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        self.redis_client_server.set(user_id, obj)
        
    def read_from_cache_user(self, user_id):
        cnt=0
        res=None
        while cnt<3 and res is None:
            try:
                res= self.__get_random_server().get(user_id)
            except:
                cnt+=1
        return res
    
    def keys(self):
        return self.__get_random_server().keys()
    def delete(self, key):
        return self.redis_client_server.delete(key)
    
cache= RedisCache()

    
    
        
    




