#!/bin/bash

octave generate.m
cat data1 data2 > temp 
shuf temp > data 
rm temp 
