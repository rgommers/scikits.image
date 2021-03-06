import numpy as np
from numpy.testing import *

from scikits.image.transform import *

def test_frt():
    SIZE = 59
    try:
        import sympy.ntheory as sn
        assert sn.isprime(SIZE) == True
    except ImportError:
        pass

    # Generate a test image
    L = np.tri(SIZE, dtype=np.int32) + np.tri(SIZE, dtype=np.int32)[::-1]
    f = frt2(L)
    fi = ifrt2(f)
    assert len(np.nonzero(L-fi)[0]) == 0
