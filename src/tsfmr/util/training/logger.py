# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
import sys
from datetime import datetime

class Logger(object):
    pending_output = ''
    def __init__(self, logfile):
        self.terminal = sys.stdout
        self.log = open(logfile, "a")
        self.newline = True
        

    def write(self, message):
        prefix = "[%s] " % str(datetime.now())
        if message == '\n':
            self.terminal.write(message)
            self.log.write(message)
            self.newline = True
        elif self.newline:
            message = message.replace("\n", "\n" + " " * len(prefix))
            to_print = "%s%s" % (prefix, message)
            self.terminal.write(to_print)
            self.log.write(to_print)
            self.newline = False
        else:
            self.terminal.write(message)
            self.log.write(message)
        self.log.flush()

    def flush(self):
        #this flush method is needed for python 3 compatibility.
        #this handles the flush command by doing nothing.
        #you might want to specify some extra behavior here.
        self.terminal.flush()
        self.log.flush()
        pass   