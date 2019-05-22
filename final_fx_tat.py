#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 21 13:20:45 2019

@author: tarun.bhavnani
"""

import os
import pandas as pd
from datetime import datetime
import calendar
import holidays
from datetime import date, timedelta
in_holidays = holidays.HolidayBase()

in_holidays.append({date(2019, 1, 1): "New Year's Day",
            date(2019, 1, 26): "Republic Day",
            date(2019, 3, 21): 'Holi',
            date(2019, 3, 6): 'Canberra Day',
            date(2019, 5, 1): 'Maharashtra Day',
            date(2019, 8, 15): 'Independence Day',
            date(2019, 9, 2): 'Ganesh Chaturthi',
            date(2019, 10, 2): 'Gandhi Jayanti',
            date(2019, 10, 8): "Dassera",
            date(2019, 10, 28): 'Diwali_Balipratipada_Padwa',
            date(2019, 10, 29): 'Diwali_Bhaiduj'})
#==========================================================================================================
year = 2017
c= calendar.Calendar()
curretyear = datetime.now().year + 1
for y in range(year,curretyear):
   for m in range(1,13):
       sat=[]
       sun=[]
       mnthcal=c.monthdatescalendar(y,m)
       for week in mnthcal:
           for day in week:
               if(day.weekday() == calendar.SATURDAY and
                   day.month == m):
                   sat.append(day)
               if(day.weekday() == calendar.SUNDAY and
                   day.month == m):
                   sun.append(day)
       in_holidays.append(sat[1:3])
       in_holidays.append(sun)
       



def tat(first, last):
 
 try:
     
  if first.hour<10:
      first=first.replace(hour=10, minute=0, second=0, microsecond=0)
  if first.hour>=18:
      first=first.replace(hour=18, minute=0, second=0, microsecond=0)
  if last.hour>=18:
      last=last.replace(hour=18, minute=0, second=0, microsecond=0)
  if last.hour<10 and last.hour!=0:
     last=last.replace(hour=10, minute=0, second=0, microsecond=0)
  if last.hour==0:
     last=last.replace(hour=18, minute=0, second=0, microsecond=0)
     
  last1=date(last.year, last.month, last.day)
  first1=date(first.year,first.month,first.day)   
  delta= last1-first1

  hrs=0
  mins=0
  for i in range(delta.days+1 ):
      print(i)
    
      x= first+timedelta(i)
      print(x)
      
      if i==0 and delta.days==0:
          if x in in_holidays:
              hours=2
              minutes=0
          else:
              td= last-first
              hours= td.components.hours
              minutes= td.components.minutes
          
          
      elif i==0 and delta.days!=0:
          if x in in_holidays:
              hours=2
              minutes=0
          
          else:
              day_end= x.replace(hour=18, minute=0, second=0, microsecond=0)
              td= day_end-x
              hours= td.components.hours
              minutes= td.components.minutes
 
      elif i == delta.days and i!=0:
          if last in in_holidays:
              hours=2
              minutes=0
          elif last.hour ==0 and last.minute==0:
              hours=8
              minutes=0
              
          else:
              day_start= x.replace(hour=10, minute=0, second=0, microsecond=0)
              td= last-day_start
              hours= td.components.hours
              minutes= td.components.minutes




      else:
          if x in in_holidays:
              hours=0
              minutes=0
          else:
              hours=8
              minutes=0
    
#      print("----------------------",i,hours, minutes)
      hrs+=hours
      mins+=minutes        

  tat= hrs+mins/60 
  return tat
 
 except:
     tat=0
     return tat

final_frame['tatCPA']=[tat(a,b) for a,b in zip(final_frame['Data_entry_date_tat'],final_frame["PD_Pending_tat"])]



dat['tat']=[tat(a,b) for a,b in zip(dat['Data_entry_date_tat'],dat["PD_Pending_tat"])]
