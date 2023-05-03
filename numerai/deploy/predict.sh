#!/usr/bin/env bash
# This script is run by the Numerai tournament server to make predictions on the
# live dataset. We will run predictions for multiple models by running the predict.py
# script multiple times with different model names. See ModelLog.md for a list of
# supported models.
python predict.py --model-name=albania --model-id=cba9d600-0be3-40fc-b867-5c1f92311c5f
python predict.py --model-name=argentinacv_no_ntr --model-id=db358e7e-d218-4ae1-82bd-e79a854322cd
python predict.py --model-name=argentinacv_ntr50p --model-id=8d683aa1-c99e-40d8-a832-2c49e31e5637
python predict.py --model-name=argentina_no_ntr --model-id=c66550bc-5fbc-428e-8691-148b0abce8bf
python predict.py --model-name=argentina_ntr50p --model-id=f275ed90-4ecb-475e-87da-d5c9e04d5ef8
