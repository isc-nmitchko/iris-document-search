import os
from getpass import getpass
from dotenv import load_dotenv

load_dotenv()

# point env var at SSLDefs if not set
if os.environ.get("ISC_SSLconfigurations", "") == "":
    os.environ["ISC_SSLconfigurations"] = "./connection/SSLDefs.ini"


def get_env_variables():
    required_env_vars = [
        "SQL_HOSTNAME",
        "SQL_PORT",
        "SQL_NAMESPACE",
        "SQL_USERNAME",
        "SQL_PASSWORD",
        "SQL_SSLCONFIG",
        "SQL_TIMEOUT",
        "SQL_SHAREDMEMORY",
        "RAG_DEVICE_MAP",
        "VLM_DEVICE_MAP",
        "MODEL_SLUG",
    ]

    env_vars = {var: os.getenv(var) for var in required_env_vars}

    missing_vars = [key for key, value in env_vars.items() if value is None]

    if missing_vars:
        raise EnvironmentError(
            f"Missing environment variables: {', '.join(missing_vars)}"
        )

    return env_vars



def get_iris_connection_settings(env: dict):
    return {
        "hostname": env["SQL_HOSTNAME"],
        "port": int(env["SQL_PORT"]),
        "namespace": env["SQL_NAMESPACE"],
        "username": env["SQL_USERNAME"],
        "password": env["SQL_PASSWORD"],
        "sslconfig": env["SQL_SSLCONFIG"],
        "timeout": int(env["SQL_TIMEOUT"]),
        "sharedmemory": "False" not in env["SQL_SHAREDMEMORY"],
    }
