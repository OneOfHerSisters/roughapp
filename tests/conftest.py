import pandas as pd
import pytest


@pytest.fixture
def table1() -> pd.DataFrame:
    """Decision table from the thesis (Table 1).

    Nine objects x1..x9, conditional attributes a,b,c,d, decision dec.
    One conflict class: {x3, x9} share (a=1,b=1,c=1,d=0) with decisions 0 and 2.
    γ = 7/9.  α(X0)=3/5=0.60, α(X1)=2/2=1.0, α(X2)=2/4=0.50.
    """
    return pd.DataFrame(
        {
            "a":   [0, 0, 1, 1, 0, 1, 0, 1, 1],
            "b":   [0, 0, 1, 1, 1, 1, 1, 0, 1],
            "c":   [0, 1, 1, 2, 0, 0, 1, 1, 1],
            "d":   [0, 1, 0, 0, 1, 2, 2, 0, 0],
            "dec": [0, 0, 0, 0, 1, 1, 2, 2, 2],
        }
    )
