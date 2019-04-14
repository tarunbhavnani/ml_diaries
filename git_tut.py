#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 13 21:09:49 2019

@author: tarun
"""

git config --global user.name "Tarun Bhavnani"
git config --global user.email "tarun.bhavnani@gmail.com"

git config --list

#start tracking
cd gitchkdir
git init
ls -la
#stop tracking
rf -rm .git


#before first commit
git status

#to ignore files
touch .gitignore

#open gitignore and add the names. eg wa.py
"""
.wa.py
"""

#add files to staging area

git add .gitignore
git status 

#add everything

git add -A

#remove 

git reset .gitignore
git status

#remove all
git reset



#commit

git commit -m "message"
git status
git log


#clone a repo and star working on it

git clone https://github.com/tarunbhavnani/ml_diaries.git .

#dot places in the current directory
#fatal: destination path '.' already exists and is not an empty directory.
git clone https://github.com/tarunbhavnani/ml_diaries.git ml_diaries
#creates a folder ml_diaroes and puts the data in side
cd ml_diaries
git remote -v
git branch -a



#do changes and update


#do some change to one of the codes

git diff

#it gives all the changes that have happened in any of the files
git status

#tells which files are updated
git add -A
#adds all to staging

git commit -m "see first change updated"

#now they are committed locally and need to be pushed in cloud repositery

git pull origin master

#tells us any changes which have been done since we updated

#push our changes to the master repositeru

git push origin master

#username password put

#create a branch

git branch new_branch
git branch

* master
  new_branch

*means working in master

git checkout new_branch

git branch
  master
* new_branch


#do some changes to "cluster_patterns.py" again

git push -u origin new_branch
git branch -a


#now this we are developing project in new branch
#when we are ready to merge


git checkout master

git pull origin master

git branch --merge

git merge new_branch

git branch --merge
#now u can see new_branch

git push origin master

#master has the new branch merged
#to check

git branch --merge

#we can delete the branch

git branch -d new_branch

#check
git branch -a

#but we still have it in the remote repositery

git push origin --delete new_branch

git branch -a


























