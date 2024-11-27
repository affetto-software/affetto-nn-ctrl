from __future__ import annotations

from io import StringIO

import numpy as np
import numpy.testing as nt
import pytest
from pyplotutil.datautil import Data

from affetto_nn_ctrl.model_utility import PreviewRef, PreviewRefParams

TOY_JOINT_DATA_TXT = """\
t,q0,q5,dq0,dq5,pa0,pa5,pb0,pb5,ca0,ca5,cb0,cb5
4.00,0.81,20.80,-2.96,-25.81,378.01,377.55,401.96,418.98,169.11,166.36,170.89,173.64
4.03,0.73,20.09,-1.60,-23.02,377.95,378.46,401.70,418.17,169.23,169.07,170.77,170.93
4.07,0.75,19.42, 1.56,-21.88,377.68,379.30,401.75,417.06,169.17,172.22,170.83,167.78
4.10,0.83,18.70, 1.28,-15.13,377.97,380.30,401.94,416.46,169.02,175.92,170.98,164.08
4.13,0.83,18.34,-0.41,-10.35,378.01,380.76,401.63,415.07,169.04,179.60,170.96,160.40
4.17,0.84,18.11, 0.88, -6.34,377.76,381.88,401.53,412.69,169.01,183.58,170.99,156.42
4.20,0.86,17.99,-0.40, -1.26,377.97,383.14,402.25,409.73,168.98,187.50,171.02,152.50
4.23,0.81,17.99,-2.04,  0.57,377.97,386.51,401.27,407.22,169.09,191.25,170.91,148.75
4.27,0.76,18.02,-1.38,  0.76,378.28,395.33,400.99,403.54,169.18,194.99,170.82,145.01
4.30,0.75,18.07, 0.56,  1.78,378.36,408.76,401.51,394.00,169.17,198.67,170.83,141.33
"""


ALIGNED_TOY_JOINT_DATA_TXT = """\
   t,  q0,   q5,  dq0,   dq5,   pa0,   pa5,   pb0,   pb5,   ca0,   ca5,   cb0,   cb5
4.00,0.81,20.80,-2.96,-25.81,378.01,377.55,401.96,418.98,169.11,166.36,170.89,173.64
4.03,0.73,20.09,-1.60,-23.02,377.95,378.46,401.70,418.17,169.23,169.07,170.77,170.93
4.07,0.75,19.42, 1.56,-21.88,377.68,379.30,401.75,417.06,169.17,172.22,170.83,167.78
4.10,0.83,18.70, 1.28,-15.13,377.97,380.30,401.94,416.46,169.02,175.92,170.98,164.08
4.13,0.83,18.34,-0.41,-10.35,378.01,380.76,401.63,415.07,169.04,179.60,170.96,160.40
4.17,0.84,18.11, 0.88, -6.34,377.76,381.88,401.53,412.69,169.01,183.58,170.99,156.42
4.20,0.86,17.99,-0.40, -1.26,377.97,383.14,402.25,409.73,168.98,187.50,171.02,152.50
4.23,0.81,17.99,-2.04,  0.57,377.97,386.51,401.27,407.22,169.09,191.25,170.91,148.75
4.27,0.76,18.02,-1.38,  0.76,378.28,395.33,400.99,403.54,169.18,194.99,170.82,145.01
4.30,0.75,18.07, 0.56,  1.78,378.36,408.76,401.51,394.00,169.17,198.67,170.83,141.33

"""


@pytest.fixture(scope="session")
def toy_joint_data() -> Data:
    return Data(StringIO(TOY_JOINT_DATA_TXT))


class TestPreviewRef:
    @pytest.fixture
    def default_adapter(self) -> PreviewRef:
        return PreviewRef(PreviewRefParams(active_joints=[5], dt=0.033, preview_step=1, include_dqdes=False))

    @pytest.mark.parametrize(
        ("params", "expected"),
        [
            (
                PreviewRefParams([5], 0.033, 1),
                """\
20.80,-25.81,377.55,418.98,20.09
20.09,-23.02,378.46,418.17,19.42
19.42,-21.88,379.30,417.06,18.70
18.70,-15.13,380.30,416.46,18.34
18.34,-10.35,380.76,415.07,18.11
18.11, -6.34,381.88,412.69,17.99
17.99, -1.26,383.14,409.73,17.99
17.99,  0.57,386.51,407.22,18.02
18.02,  0.76,395.33,403.54,18.07
""",
            ),
            (
                PreviewRefParams([5], 0.033, 3),
                """\
20.80,-25.81,377.55,418.98,18.70
20.09,-23.02,378.46,418.17,18.34
19.42,-21.88,379.30,417.06,18.11
18.70,-15.13,380.30,416.46,17.99
18.34,-10.35,380.76,415.07,17.99
18.11, -6.34,381.88,412.69,18.02
17.99, -1.26,383.14,409.73,18.07
""",
            ),
            (
                PreviewRefParams([5], 0.033, 7),
                """\
20.80,-25.81,377.55,418.98,17.99
20.09,-23.02,378.46,418.17,18.02
19.42,-21.88,379.30,417.06,18.07
""",
            ),
        ],
    )
    def test_make_feature(self, toy_joint_data: Data, params: PreviewRefParams, expected: str) -> None:
        adapter = PreviewRef(params)
        feature = adapter.make_feature(toy_joint_data)
        expected_feature = np.loadtxt(StringIO(expected), delimiter=",")
        nt.assert_array_equal(feature, expected_feature)

    @pytest.mark.parametrize(
        ("params", "expected"),
        [
            (
                PreviewRefParams([5], 0.033, 1),
                """\
166.36,173.64
169.07,170.93
172.22,167.78
175.92,164.08
179.60,160.40
183.58,156.42
187.50,152.50
191.25,148.75
194.99,145.01
""",
            ),
            (
                PreviewRefParams([5], 0.033, 3),
                """\
166.36,173.64
169.07,170.93
172.22,167.78
175.92,164.08
179.60,160.40
183.58,156.42
187.50,152.50
""",
            ),
            (
                PreviewRefParams([5], 0.033, 7),
                """\
166.36,173.64
169.07,170.93
172.22,167.78
""",
            ),
        ],
    )
    def test_make_target(self, toy_joint_data: Data, params: PreviewRefParams, expected: str) -> None:
        adapter = PreviewRef(params)
        target = adapter.make_target(toy_joint_data)
        expected_target = np.loadtxt(StringIO(expected), delimiter=",")
        nt.assert_array_equal(target, expected_target)

    @pytest.mark.parametrize(
        "params",
        [
            PreviewRefParams([5], 0.033, 1),
            PreviewRefParams([5], 0.033, 3),
            PreviewRefParams([5], 0.033, 7),
        ],
    )
    def test_make_model_input(self, params: PreviewRefParams) -> None:
        dof = 6
        rng = np.random.default_rng()
        q = rng.uniform(15, 25, size=dof)
        dq = rng.uniform(-30, 30, size=dof)
        pa = rng.uniform(300, 400, size=dof)
        pb = rng.uniform(400, 500, size=dof)

        def qdes(t: float) -> np.ndarray:
            return q - t

        t = rng.uniform(4, 6)
        adapter = PreviewRef(params)
        x = adapter.make_model_input(t, {"q": q, "dq": dq, "pa": pa, "pb": pb}, {"qdes": qdes})
        i = adapter.params.active_joints[0]
        offset = params.preview_step * params.dt
        expected = np.array([[q[i], dq[i], pa[i], pb[i], qdes(t + offset)[i]]], dtype=float)
        nt.assert_array_equal(x, expected)

    @pytest.mark.parametrize(
        "params",
        [
            PreviewRefParams([5], 0.033, 1),
            PreviewRefParams([5], 0.033, 3),
            PreviewRefParams([5], 0.033, 7),
        ],
    )
    def test_make_ctrl_input(self, params: PreviewRefParams) -> None:
        dof = 6
        rng = np.random.default_rng()
        base_ca = rng.uniform(170, 200, size=dof)
        base_cb = rng.uniform(140, 170, size=dof)
        y = np.array([[rng.uniform(170, 200), rng.uniform(140, 170)]])

        adapter = PreviewRef(params)
        ca, cb = adapter.make_ctrl_input(y, {"ca": base_ca, "cb": base_cb})

        i = adapter.params.active_joints[0]
        expected_ca = base_ca.copy()
        expected_ca[i] = y[0][0]
        expected_cb = base_cb.copy()
        expected_cb[i] = y[0][1]
        nt.assert_array_equal(ca, expected_ca)
        nt.assert_array_equal(cb, expected_cb)


# Local Variables:
# jinx-local-words: "cb ctrl dq params pb qdes"
# End:
