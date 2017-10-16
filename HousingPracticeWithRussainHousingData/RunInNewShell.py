from sys import executable
from subprocess import Popen, CREATE_NEW_CONSOLE

Popen([executable, 'DataExperiment_19_06_17.py'], creationflags=CREATE_NEW_CONSOLE)

input('Enter to exit from this launcher script...')
