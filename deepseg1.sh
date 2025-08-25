PATH=$(dirname $0)/.venv/bin/:$(dirname $0)/venv/bin/:$PATH
python3 $(dirname $0)/hippodeep.py $@
