import logging
from pathlib import Path
from typing import Any
import numpy as np
import scipy as sp
import subprocess
import struct
import toml
from Messaging import TwoWayPipe, MessageType, Message, exit_msg, trigger_save_msg
from pytams.fmodel import ForwardModelBaseClass
from pytams.tams import TAMS

_logger = logging.getLogger(__name__)


class Boussinesq2DModelCpp(ForwardModelBaseClass):
    """A forward model for the 2D Boussinesq C++ model.

    This variant of the Boussinesq model uses a C++ executable to
    advance the model instead of a python implementation. This demonstrates one way of
    coupling TAMS to existing external software.
    """

    def _init_model(self, params: dict | None = None, ioprefix: str | None = None) -> None:
        """Initialize the model."""
        self._ioprefix = ioprefix

        # Parse parameters
        subparms = params.get("model", {})
        self._M = subparms.get("size_M", 40)  # Horizontals
        self._N = subparms.get("size_N", 80)  # Verticals
        self._K = subparms.get("K", 4)  # Number of forcing modes = 2*K
        self._eps = subparms.get("epsilon", 0.05)  # Noise level
        self._exec = subparms.get("exec", None)
        self._exec_cmd = [self._exec,
                          "-M", str(self._M),
                          "-N", str(self._N),
                          "-K", str(self._K),
                          "--eps", str(self._eps),
                          "--pipe_id", str(ioprefix)]

        # A handle to the C++ subprocess and the twoway pipe
        self._proc = None
        self._pipe = None
        #self._pipe = TwoWayPipe(self._ioprefix)

        # Initialize random number generator
        # If deterministic run, set seed from the traj id
        if subparms["deterministic"]:
            self._rng = np.random.default_rng(int(ioprefix[4:10]))
        else:
            self._rng = np.random.default_rng()

        self._db_path = self._workdir.parents[1]
        if not self._workdir.exists():
            self._workdir.mkdir()

        # The state is a path to a npy file on disk
        self._state = subparms.get("init_state", None)

    @classmethod
    def name(cls) -> str:
        """Return the model name."""
        return "2DBoussinesqModelCpp"

    def get_current_state(self) -> Any:
        """Access the model state."""
        return self._state

    def get_last_statefile(self) -> str | None:
        if not self._proc:
            return None

        self._pipe.post_message(Message(MessageType.GETSTATE))

        ret = self._pipe.get_message()
        assert ret.mtype == MessageType.DONE

        return ret.data.decode("utf-8")

    def set_current_state(self, state: Any) -> None:
        """Set the model state.

        And transfer it to the C++ process if opened

        Args:
            state: the new state
        """
        self._state = state
        if not self._proc:
            return

        self._pipe.post_message(Message(MessageType.SETSTATE, data = state.encode("utf-8")))

        ret = self._pipe.get_message()
        assert ret.mtype == MessageType.DONE



    def _advance(self, step: int, time: float, dt: float, noise: Any, need_end_state: bool) -> float:
        """Advance the model.

        Args:
            step: the current step counter
            time: the starting time of the advance call
            dt: the time step size over which to advance
            noise: the noise to be used in the model step
            need_end_state: whether the step end state is needed

        Return:
            Some model will not do exactly dt (e.g. sub-stepping) return the actual dt
        """
        if not self._proc:
            # Initialize the C++ process and the twoway pipe
            self._proc = subprocess.Popen(self._exec_cmd)

            self._pipe = TwoWayPipe(self._ioprefix)
            self._pipe.open()

            # Send the workdir
            self._pipe.post_message(Message(MessageType.SETWORKDIR, data = self._workdir.as_posix().encode("utf-8")))

            # Set the initial state
            self.set_current_state(self._state)

        if need_end_state:
            self._pipe.post_message(trigger_save_msg)

        self._pipe.post_message(Message(MessageType.ONESTOCHSTEP, data = noise.tobytes()))
        ret = self._pipe.get_message()
        assert ret.mtype == MessageType.DONE
        self._score = struct.unpack("d", ret.data)[0]

        if need_end_state:
            self._state = self.get_last_statefile()
        else:
            self._state = None

        return dt

    def _clear_model(self) -> None:
        if self._proc:
            self._pipe.post_message(exit_msg)
            self._proc.wait()
            self._proc = None
            self._pipe.close()
            self._pipe = None

    def score(self) -> float:
        """Compute the score function.

        The current score function is a nomalized distance between the ON
        and OFF states in the stream function space (specifically the
        mean streamfunction in the southern ocean).

        Return:
            the score
        """
        if not self._proc:
            return 0.0

        self._pipe.post_message(Message(MessageType.GETSCORE))
        ret = self._pipe.get_message()
        assert ret.mtype == MessageType.DONE
        score = struct.unpack("d", ret.data)[0]
        return score

    def make_noise(self) -> Any:
        """Return a random noise."""
        return self._rng.normal(0, 1, size=(2 * self._K))


if __name__ == "__main__":
    fmodel = Boussinesq2DModelCpp
    tams = TAMS(fmodel_t=fmodel)
    transition_proba = tams.compute_probability()
    print(f"Transition probability: {transition_proba}")

    #params = toml.load("input.toml")
    #
    #model = Boussinesq2DModelCpp(params, "000000")
    #model.advance(0.005, False)
    #print(model.score())
    #model.advance(0.005, False)
    #print(model.score())
    #model.advance(0.005, True)
    #print(model.get_last_statefile())
