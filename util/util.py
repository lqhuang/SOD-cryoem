import shlex
import subprocess
# import sklearn.preprocessing as prep


# def standard_scale(X_train, X_test):
#     preprocessor = prep.StandardScaler().fit(X_train)
#     X_train = preprocessor.transform(X_train)
#     X_test = preprocessor.transform(X_test)
#     return X_train, X_test


def run_system_command(command_string):
    """Function used to run the system command and return the log"""
    process = subprocess.Popen(shlex.split(command_string),
                               stdout=subprocess.PIPE)  # Run system command
    output, _ = process.communicate()  # Get the log.
    return output.decode('utf-8')  # return the log file
