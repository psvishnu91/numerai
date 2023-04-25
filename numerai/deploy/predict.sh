#!/usr/bin/env bash
# This script is run by the Numerai tournament server to make predictions on the
# live dataset. We will run predictions for multiple models by running the predict.py
# script multiple times with different model names. See ModelLog.md for a list of
# supported models.
python predict.py --model-name=albania
