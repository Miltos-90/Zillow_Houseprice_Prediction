#!/bin/bash
# Script to setup a standalone spark cluster.
# Run with the following command: sudo -E ./setup.sh

# Make sure that the environment variable SPARK_HOME has been set (=/opt/spark for Ubuntu)
if [ -z "${SPARK_HOME}" ]
then
    echo "ERROR: SPARK_HOME environment variable is not set."
    exit 1
fi

CONF_FILE=${SPARK_HOME}/conf/spark-env.sh
#CONF_FILE=./test.txt
GPU_DISCOVERY_SOURCE_URL=https://raw.githubusercontent.com/apache/spark/master/examples/src/main/scripts/getGpusResources.sh
GPU_DISCOVERY_TARGET_FILE=/opt/sparkRapidsPlugin/getGpusResources.sh

# Settings for the configuration file (per worker). Only used if the configuration file (CONF_FILE) does not already exist.
NUM_WORKERS=1
NUM_CORES=10
NUM_GPUS=1
MEMORY=30g

# Check which files already exist, so that they will not be modified
if [ -e ${CONF_FILE} ]; then
    CONF_FILE_EXISTS=1
    echo "Found existing configuration file."
else
    CONF_FILE_EXISTS=0
fi

NEED_GPU=$(( ${NUM_GPUS} > 0 ? 1 : 0 ))

if [ ${NEED_GPU} ]; then
    if [ ! -e ${GPU_DISCOVERY_TARGET_FILE} ]; then
        GPU_FILE_EXISTS=0
    else
        echo "Found existing GPU discovery script."
        GPU_FILE_EXISTS=1
    fi
fi


# Make configuration file if it does not exist
if [ ${CONF_FILE_EXISTS} == 0 ]; then
    echo "Generating configuration file."
    
    CONF_TEXT="
    SPARK_WORKER_INSTANCES=${NUM_WORKERS}\n
    SPARK_WORKER_CORES=${NUM_CORES}\n
    SPARK_WORKER_MEMORY=${MEMORY}
    "

    echo -e ${CONF_TEXT} >> ${CONF_FILE}

    if [ ${NEED_GPU} ]; then
        GPU_WORKER_OPTS="SPARK_WORKER_OPTS=\"-Dspark.worker.resource.gpu.amount=${NUM_GPUS} -Dspark.worker.resource.gpu.discoveryScript=${GPU_DISCOVERY_TARGET_FILE}\""
        echo -e ${GPU_WORKER_OPTS} >> ${CONF_FILE}
    fi

fi

# Setup GPU discovery script
if [ ${GPU_FILE_EXISTS} == 0 ]; then     
    if [ ! -x ${wget} ]; then       # Check if the 'wget' command is available to download the file
        echo "ERROR: No wget." >&2 
        exit 1
    else
        echo "Generating GPU Discovery script."
        wget -q -O ${GPU_DISCOVERY_TARGET_FILE} ${GPU_DISCOVERY_SOURCE_URL}
        sudo chmod +x ${GPU_DISCOVERY_TARGET_FILE}
    fi 
fi