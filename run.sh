#!/bin/bash

# CHAINLIT_HOST and CHAINLIT_PORT
# --host and --port when running chainlit run ....
# nohup ./run.sh &> chainlit-chatbot.log &
# nohup chainlit run app.py &> chainlit-chatbot.log &
chainlit run chatbot.py --host 0.0.0.0 --port $PORT
