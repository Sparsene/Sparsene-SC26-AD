#!/bin/bash
GlogPath="${DTC_HOME}/third_party/glog"
if [ -z "$GlogPath" ]
then
	echo "Defining the GLOG path is necessary, but it has not been defined."
else
	export GLOG_PATH=$GlogPath
	export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$GLOG_PATH/build/lib
	export CPLUS_INCLUDE_PATH=$CPLUS_INCLUDE_PATH:$GLOG_PATH/build/include
	export LIBRARY_PATH=$LD_LIBRARY_PATH:$GLOG_PATH/build/lib
fi


SputnikPath="${DTC_HOME}/third_party/sputnik"
if [ -z "$SputnikPath" ]
then
	echo "Defining the Sputnik path is necessary, but it has not been defined."
else
	export SPUTNIK_PATH=$SputnikPath
	export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$SPUTNIK_PATH/build/sputnik
fi

