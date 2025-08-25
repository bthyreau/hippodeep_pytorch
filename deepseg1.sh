#!/bin/bash
r0=$(realpath $0)
PATH=$(dirname $r0)/.venv/bin/:$(dirname $r0)/venv/bin/:$PATH
python3 $(dirname $r0)/hippodeep.py $@
