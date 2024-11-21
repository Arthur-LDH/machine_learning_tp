#!/usr/bin/env python3
import logging

# Get logger (def level: INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Get console handler
# On log tout ce qui est possible ici (i.e >= level du logger)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)

# Manage formatter
formatter = logging.Formatter("[%(asctime)s] - %(name)s.%(funcName)s() - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
ch.setFormatter(formatter)

# Add handler to the logger
logger.addHandler(ch)