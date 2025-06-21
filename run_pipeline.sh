#!/bin/bash

# Optional: activate your virtualenv here
# source venv/bin/activate

COMMAND=$1
START_DATE=${2:-None}


case $COMMAND in
  pipeline)
    echo "Running pipeline with start_date=${START_DATE}"
    python3 pipeline.py "${START_DATE}"
    ;;

  train)
    echo "Training stock model..."
    python3 -c "from ml.training import train_stock_model; train_stock_model()"
    ;;

  incremental)
    echo "Running incremental training..."
    python3 -c "from ml.training import incremental_train; incremental_train()"
    ;;

  *)
    echo "Usage: $0 [pipeline <start_date>] | [train] | [incremental]"
    exit 1
    ;;
esac
