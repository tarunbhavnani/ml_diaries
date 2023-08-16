# -*- coding: utf-8 -*-
"""
Created on Wed Aug 16 11:01:17 2023

@author: tarun
"""

class CacheInterface():
    """
    interface for interactting with cache
    """
    def ping(self):
        pass
    
    def write_to_cache(self, user_id, obj):
        """
        

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
        pass
    def read_from_cache_user(self, user_id):
        """
        

        Parameters
        ----------
        user_id : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        pass
    