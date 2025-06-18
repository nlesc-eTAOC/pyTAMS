"""A TwoWayPipe class to communicate with C++ code."""
import os
import errno
import subprocess
import time
import sys
import struct
import numpy as np
from typing import Any
from enum import Enum

class MessageType(Enum):
    NULL = 0
    SETWORKDIR = 1
    SETSTATE = 2
    GETSTATE = 3
    GETSCORE = 4
    ONESTEP = 5
    ONESTOCHSTEP = 6
    STATESAVE = 7
    DONE = 8
    EXIT = 9

class Message():
    mtype : MessageType = MessageType.NULL
    size : int = 0
    data : bytes | None = None

    def __init__(self,
                 mtype : int,
                 msize : int = -1,
                 data : Any = None):
        self.mtype = mtype
        if data is not None:
            self.data = data
            self.size = len(data)
            if msize > 0:
                assert msize == self.size
        else:
            if msize > 0:
                self.size = msize
            else:
                self.size = 0

onestep_msg = Message(MessageType.ONESTEP)
exit_msg = Message(MessageType.EXIT)
trigger_save_msg = Message(MessageType.STATESAVE)

class TwoWayPipe():
    # Two pipes: one for C++ to Python and one for Python to C++
    pipe_wr = None
    pipe_rd = None

    def __init__(self, idstr):

        self.pipe_wr = './ptoc_' + idstr
        self.pipe_rd = './ctop_' + idstr

        try:
          os.mkfifo(self.pipe_wr, mode=0o777)
        except OSError as oe:
          if oe.errno != errno.EEXIST:
            raise

        try:
          os.mkfifo(self.pipe_rd, mode=0o777)
        except OSError as oe:
          if oe.errno != errno.EEXIST:
            raise

        self.frd = None
        self.fwr = None

    def open(self):
        # Open the read pipe, locked until C++ code open its write pipe
        self.frd = open(self.pipe_rd, 'rb')
        # Then open the write pipe, unlocking the C++ read pipe
        self.fwr = open(self.pipe_wr, 'wb')

    def close(self):
        if self.frd is not None: self.frd.close()
        if self.fwr is not None: self.fwr.close()

    def __del__(self):
        self.close()
        if os.path.exists(self.pipe_wr): os.remove(self.pipe_wr)
        if os.path.exists(self.pipe_rd): os.remove(self.pipe_rd)

    def get_message(self) -> Message:
        raw = self.frd.read(4)
        mtype = struct.unpack('i', raw)[0]
        raw = self.frd.read(4)
        size = struct.unpack('i', raw)[0]
        msg = Message(MessageType(mtype), msize = int(size))
        if msg.size > 0:
            msg.data = self.frd.read(msg.size)

        return msg

    def post_message(self, msg: Message):
        self.fwr.write(struct.pack('i', msg.mtype.value))
        self.fwr.flush()

        self.fwr.write(struct.pack('i', msg.size))
        self.fwr.flush()

        if msg.size > 0:
            self.fwr.write(msg.data)
            self.fwr.flush()
