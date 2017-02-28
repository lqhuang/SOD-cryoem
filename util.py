import subprocess

import numpy as np

def run_system_command(command_string):
    """Function used to run the system command and return the log"""
    process = subprocess.Popen(command_string, stdout=subprocess.PIPE, shell=True) # Run system command
    output = process.communicate()  # Get the log.
    return output[0]  # return the log file