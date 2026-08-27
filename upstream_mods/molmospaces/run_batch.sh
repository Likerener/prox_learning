#!/bin/bash

for i in {1..8}

do
	echo "Run $i"
	python molmo_spaces/data_generation/main.py FrankaPickOmniCamConfig
done
