#!/usr/bin/env bash

python -m grpc_tools.protoc -Iapi --python_out=src/gen-proto --grpc_python_out=src/gen-proto api/invigilator.proto
python -m grpc_tools.protoc -Iapi --python_out=src/gen-proto --grpc_python_out=src/gen-proto api/signUpIn.proto
python -m grpc_tools.protoc -Iapi --python_out=src/gen-proto --grpc_python_out=src/gen-proto api/student.proto