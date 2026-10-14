import os
import tempfile

# Must run before solarsense is imported: isolate artifacts and force the offline (no-LLM) path.
os.environ["SOLARSENSE_ARTIFACT_DIR"] = tempfile.mkdtemp()
os.environ["SOLARSENSE_GROQ_API_KEY"] = ""
os.environ["GROQ_API_KEY"] = ""
os.environ["SOLARSENSE_ADMIN_API_KEY"] = "test-admin-key"

import pytest  # noqa: E402

from solarsense.data.simulator import simulate  # noqa: E402
from solarsense.monitoring.anomaly import PerformanceModel  # noqa: E402


@pytest.fixture(scope="session")
def df():
    return simulate()


@pytest.fixture(scope="session")
def perf(df):
    return PerformanceModel.calibrate(df, 5000.0)
