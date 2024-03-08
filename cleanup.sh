# Run with: sudo -E ./cleanup.sh
rm ${SPARK_HOME}/conf/spark-env.sh
rm -rf ${SPARK_HOME}/logs/*
rm /opt/sparkRapidsPlugin/getGpusResources.sh
rm ./mlruns.sqlite
rm -rf ./spark-warehouse
rm -rf ./zillow_houseprice_prediction/artifacts
rm -rf ./zillow_houseprice_prediction/mlruns