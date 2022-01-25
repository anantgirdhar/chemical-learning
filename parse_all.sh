#!/bin/sh
find RMG-database-1268e4bb446736a76370418f0bebf22e006cf6a6/input/kinetics/libraries/ -type d -maxdepth 1 -exec python parse_rmg_kinetics_library.py {} \;
