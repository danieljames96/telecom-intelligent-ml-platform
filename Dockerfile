FROM jupyter/pyspark-notebook:latest

# Use Java 11 to avoid Subject.getSubject error
ENV JAVA_HOME=/usr/lib/jvm/java-11-openjdk
ENV PATH=$JAVA_HOME/bin:$PATH

# Install OpenJDK 11
USER root
RUN apt-get update && apt-get install -y openjdk-11-jdk && apt-get clean

# Switch back to default notebook user
USER jovyan
