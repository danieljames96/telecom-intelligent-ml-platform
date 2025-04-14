FROM jupyter/pyspark-notebook:latest

USER root

# Install Java 11
RUN apt-get update && apt-get install -y openjdk-11-jdk && apt-get clean

# Set Java 11 as default (important!)
ENV JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
ENV PATH=$JAVA_HOME/bin:$PATH

# Also update alternatives so it's default system-wide
RUN update-alternatives --set java $JAVA_HOME/bin/java

USER jovyan