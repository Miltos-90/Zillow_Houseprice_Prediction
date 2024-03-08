#!/bin/bash
# Script to start a standalone spark cluster + MLFlow for experiment tracking.
MASTER_HOST=localhost
MASTER_PORT=7078
MASTER_WEBUI_PORT=8081
MLFLOW_PORT=8080
MLFLOW_STORE=sqlite:///mlruns.sqlite

# Make sure that the environment variable SPARK_HOME has been set (=/opt/spark for Ubuntu)
if [ -z "${SPARK_HOME}" ]
then
    echo "ERROR: SPARK_HOME environment variable is not set."
    exit 1
fi

# Start master node
${SPARK_HOME}/sbin/start-master.sh --host ${MASTER_HOST} --port ${MASTER_PORT} --webui-port ${MASTER_WEBUI_PORT}

# Start workers
${SPARK_HOME}/sbin/start-worker.sh spark://${MASTER_HOST}:${MASTER_PORT}

# Start mlflow
mlflow server --host ${MASTER_HOST} --port ${MLFLOW_PORT} --backend-store-uri ${MLFLOW_STORE}