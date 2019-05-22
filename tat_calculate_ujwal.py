#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 20 14:59:33 2019

@author: tarun.bhavnani
"""
/
import os
os.getcwd()
os.chdir("/home/tarun.bhavnani/Desktop/ujwal")
os.listdir()

import pandas as pd

dat= pd.read_excel('TAT20May2019_01_54PMBusiness.xlsx')

dat['Data_entry_date_tat']
srt_datetime=dat['PD_Pending_tat'][17]


from datetime import datetime

#timestamp = Timestamp('2019-05-20 12:55:09.175000')
dt_object = datetime.fromtimestamp(dat['PD_Pending_tat'][17])

print("dt_object =", dt_object)
print("type(dt_object) =", type(dt_object))


yu=pd.to_datetime(srt_datetime, format = "%d-%m-%Y %H:%M:%S %p")


##holliday
!pip install holidays
import holidays
in_holidays = holidays.HolidayBase()



from datetime import date
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
       



holi=[i for i in in_holidays]


"""#conditions

1) between dates all hours
2) one day is 8 hours
3) if first, second or last is holiday then 2 hours
4) also dynamic hours on first ad last day.
5) no hours for holidays

"""

###################################################################################
###################################################################################

from datetime import date, timedelta

#start date
a=dat['Data_entry_date_tat'][38088]
b=dat["PD_Pending_tat"][38088]

a=dat['Data_entry_date_tat'][28176]
b=dat["PD_Pending_tat"][28176]

a=dat['Data_entry_date_tat'][26235]
b=dat["PD_Pending_tat"][26235]


a=dat['Data_entry_date_tat'][26274]
b=dat["PD_Pending_tat"][26274]

a=dat['Data_entry_date_tat'][6516]
b=dat["PD_Pending_tat"][6516]

a=dat['Data_entry_date_tat'][26002]
b=dat["PD_Pending_tat"][26002]


#diff
delta= b-a



#holliday=[date(2008,8,17),date(2008,8,18),date(2008,8,19)]

# timedelta
#dt=[]
hrs=0
mins=0
for i in range(delta.days+2 ):
    print(i)
    
    x= a+timedelta(i)
    print(x)
    
    if i==0:
        print("1st day")
        if x in in_holidays:
            print("1st day holiday")
            hours=2
            minutes=0
        else:
            hours= min(17-a.hour,8)
            if hours<8:
                minutes= 60-a.minute
            else:
                minutes=0
    
    elif i==1:
        print("2nd day")
        if a in in_holidays and x in in_holidays:
            pribt("1st and 2nd day both holiday")
            hours=2
            minutes=0
        
        elif a not in in_holidays and x in in_holidays:
            print("holiday")
            hours=0
            minutes=0
        else:
            hours= 8
            minutes= 0
    
    elif i == delta.days:
        print("last day")
        if b in in_holidays:
            print("last day is holiday")
            hours=2
            minutes=0
        else:
            hours= max(b.hour-10,0)
            if hours>0:
                minutes= b.minute
            else:
                minutes=0
    else:
        if x in in_holidays:
            print("holiday")
            hours=0
            minutes=0
        else:
            hours=8
            minutes=0
    
    print("----------------------",i,hours, minutes)
    hrs+=hours
    mins+=minutes        


#hrs= hrs+round(mins/60)
#mins=mins%60

print("TAT is {} hours {} minutes".format(hrs,mins))    
    
    
    

       