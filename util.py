import subprocess
import sklearn.preprocessing as prep

def standard_scale(X_train, X_test):
    preprocessor = prep.StandardScaler().fit(X_train)
    X_train = preprocessor.transform(X_train)
    X_test = preprocessor.transform(X_test)
    return X_train, X_test

def run_system_command(command_string):
    """Function used to run the system command and return the log"""
    process = subprocess.Popen(command_string, stdout=subprocess.PIPE, shell=True) # Run system command
    output = process.communicate()  # Get the log.
    return output[0]  # return the log file