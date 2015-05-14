#!/bin/bash
cd $(dirname "$0") && pwd
ipython notebook --pylab inline
