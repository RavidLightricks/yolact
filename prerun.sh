#!/usr/bin/env bash
cd external/DCNv2
python setup.py build develop
cd ~/notebooks/
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1t4BXWwKjTlqOQCIaTRlWYcT1pPhaZRUy' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1t4BXWwKjTlqOQCIaTRlWYcT1pPhaZRUy" -O yolact_plus_resnet50_54_800000.pth && rm -rf /tmp/cookies.txt
mkdir models
mv yolact_plus_resnet50_54_800000.pth models
cnvrg clone https://app.cnvrg.lightricks.com/Lightricks/datasets/facetune-videos