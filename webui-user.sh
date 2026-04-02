#!/bin/bash

export COMMANDLINE_ARGS="--use-mps --no-half-vae --opt-sdp-attention --skip-torch-cuda-test --lowvram --theme dark --precision full"
export PYTORCH_ENABLE_MPS_FALLBACK=1
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
