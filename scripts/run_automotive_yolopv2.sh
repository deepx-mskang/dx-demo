#!/bin/bash

./kill_automotive.sh

cd ~/dx-demos/automotive/yolopv2

./build/yolopv2_async --loop --color 2 --exit-btn
