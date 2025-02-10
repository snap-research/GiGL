#!/bin/bash
set -e

is_running_on_mac() {
    [ "$(uname)" == "Darwin" ]
    return $?
}

if is_running_on_mac;
then
    echo "Setting up Scala Deps for Mac environment"
    brew install coursier/formulas/coursier && cs setup
    brew install sbt
else
    echo "Setting up Scala Deps for Linux Environment"
    mkdir -p tools/scala/coursier
    curl -fL https://github.com/coursier/coursier/releases/latest/download/cs-x86_64-pc-linux.gz | gzip -d > tools/scala/coursier/cs && chmod +x tools/scala/coursier/cs && tools/scala/coursier/cs setup -y
fi

source ~/.profile

echo "Setting up required tooling"
# Install tools needed to run spark/scala code
mkdir -p tools/scalapbc
mkdir -p tools/scala

echo "Installing tooling for spark 3.1"
gsutil cp gs://gigl-public/tools/scala/spark/spark-3.1.3-bin-hadoop3.2.tgz tools/scala/spark-3.1.3-bin-hadoop3.2.tgz
gunzip -c tools/scala/spark-3.1.3-bin-hadoop3.2.tgz | tar xopf - -C tools/scala
# Pulls custom package which allows for parsing and output tf records
gsutil cp gs://gigl-public/tools/scala/spark_packages/spark-custom-tfrecord_2.12-0.5.1.jar tools/scala/spark_packages/spark-custom-tfrecord_2.12-0.5.1.jar


echo "Installing tooling for spark 3.5; this will deprecate regular installation for spark 3.1 above"
gsutil cp gs://gigl-public/tools/scala/spark/spark-3.5.0-bin-hadoop3.tgz tools/scala/spark-3.5.0-bin-hadoop3.tgz
gunzip -c tools/scala/spark-3.5.0-bin-hadoop3.tgz | tar xopf - -C tools/scala
# Pulls custom package which allows for parsing and output tf records
gsutil cp gs://gigl-public/tools/scala/registry/spark_3.5.0-custom-tfrecord_2.12-0.6.1.jar tools/scala/spark_packages/spark-custom-tfrecord_2.12-0.5.1.jar


echo "Installing tooling for scala protobuf"
# Commenting out as we are seeing some issues in the builders downloading this from github
# curl -L -o tools/scalapbc/scalapbc.zip "https://github.com/scalapb/ScalaPB/releases/download/v0.11.11/scalapbc-0.11.11.zip"
gsutil cp gs://gigl-public/tools/scala/scalapbc/scalapbc-0.11.11.zip tools/scalapbc/scalapbc.zip
unzip -o tools/scalapbc/scalapbc.zip -d tools/scalapbc
rm tools/scalapbc/scalapbc.zip
# (svij-sc) scala35 support (this will eventually deprecate above)
gsutil cp gs://gigl-public/tools/scala/scalapbc/scalapbc-0.11.14.zip tools/scalapbc/scalapbc.zip
unzip -o tools/scalapbc/scalapbc.zip -d tools/scalapbc
rm tools/scalapbc/scalapbc.zip

# if running into the following error when running unittest locally
# `java.lang.NoClassDefFoundError: Could not initialize class org.apache.spark.storage.StorageUtils$`
# export JAVA_OPTS='--add-exports java.base/sun.nio.ch=ALL-UNNAMED'

echo "Finished installation"