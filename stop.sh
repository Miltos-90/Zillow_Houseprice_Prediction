#!/bin/bash
# Script to stop a Spark standalone cluster
${SPARK_HOME}/sbin/stop-worker.sh
${SPARK_HOME}/sbin/stop-master.sh