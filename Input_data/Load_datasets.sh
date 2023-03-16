#!/bin/bash
#Load Golub dataset from kaggle
kaggle datasets download -d crawford/gene-expression
#Unzip
unzip gene-expression.zip -d ./Input_data
#Format data-set
