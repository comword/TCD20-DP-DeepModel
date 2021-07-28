#!/usr/bin/env bash

python -m grpc_tools.protoc -Iapi --python_out=src/gen_proto --grpc_python_out=src/gen_proto api/invigilator.proto
python -m grpc_tools.protoc -Iapi --python_out=src/gen_proto --grpc_python_out=src/gen_proto api/signUpIn.proto
python -m grpc_tools.protoc -Iapi --python_out=src/gen_proto --grpc_python_out=src/gen_proto api/student.proto