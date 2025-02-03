#!/bin/bash

echo "Python search path:"
python -c "import sys; print(\"\n\".join(sys.path))"

echo -e "\nInstalled packages:"
pip list

echo -e "\nStarting application..."
exec uvicorn main:app --host 0.0.0.0 --port 8000