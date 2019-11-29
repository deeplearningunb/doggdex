from python:3.7

workdir /doggdex

copy . /doggdex

run pip install -r requirements.txt
