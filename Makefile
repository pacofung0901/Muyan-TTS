BASE_DIR := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))

.PHONY: all build clean

all: clean build

build:
	mkdir -p logs
	pip install -r requirements.txt
	git clone --branch v0.9.2 https://github.com/hiyouga/LLaMA-Factory.git $(BASE_DIR)llama-factory
	cd $(BASE_DIR)/llama-factory && pip install --no-deps -e .
	
clean:
	rm -rf $(BASE_DIR)llama-factory