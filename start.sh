#!/bin/bash
# start.sh

# Run the FastAPI server using Uvicorn
uvicorn app.main:app --host 0.0.0.0 --port 10000
