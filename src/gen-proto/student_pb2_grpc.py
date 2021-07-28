# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
import grpc

import student_pb2 as student__pb2


class StudentAppStub(object):
  # missing associated documentation comment in .proto file
  pass

  def __init__(self, channel):
    """Constructor.

    Args:
      channel: A grpc.Channel.
    """
    self.UpPredictResult = channel.unary_unary(
        '/student.StudentApp/UpPredictResult',
        request_serializer=student__pb2.ModelPredict.SerializeToString,
        response_deserializer=student__pb2.CommonGetResponse.FromString,
        )
    self.GetExams = channel.unary_unary(
        '/student.StudentApp/GetExams',
        request_serializer=student__pb2.CommonGetRequest.SerializeToString,
        response_deserializer=student__pb2.ExamResponse.FromString,
        )
    self.GetPredicts = channel.unary_unary(
        '/student.StudentApp/GetPredicts',
        request_serializer=student__pb2.GetPredictRequest.SerializeToString,
        response_deserializer=student__pb2.GetPredictResponse.FromString,
        )
    self.StreamVideo = channel.stream_stream(
        '/student.StudentApp/StreamVideo',
        request_serializer=student__pb2.StreamVideoRequest.SerializeToString,
        response_deserializer=student__pb2.CommonGetResponse.FromString,
        )
    self.GetUserDetail = channel.unary_unary(
        '/student.StudentApp/GetUserDetail',
        request_serializer=student__pb2.CommonGetRequest.SerializeToString,
        response_deserializer=student__pb2.CommonGetResponse.FromString,
        )
    self.PutUserDetail = channel.unary_unary(
        '/student.StudentApp/PutUserDetail',
        request_serializer=student__pb2.CommonGetRequest.SerializeToString,
        response_deserializer=student__pb2.CommonGetResponse.FromString,
        )


class StudentAppServicer(object):
  # missing associated documentation comment in .proto file
  pass

  def UpPredictResult(self, request, context):
    """return predictId
    """
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')

  def GetExams(self, request, context):
    # missing associated documentation comment in .proto file
    pass
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')

  def GetPredicts(self, request, context):
    # missing associated documentation comment in .proto file
    pass
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')

  def StreamVideo(self, request_iterator, context):
    # missing associated documentation comment in .proto file
    pass
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')

  def GetUserDetail(self, request, context):
    """extensible JSON user details in msg
    """
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')

  def PutUserDetail(self, request, context):
    # missing associated documentation comment in .proto file
    pass
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')


def add_StudentAppServicer_to_server(servicer, server):
  rpc_method_handlers = {
      'UpPredictResult': grpc.unary_unary_rpc_method_handler(
          servicer.UpPredictResult,
          request_deserializer=student__pb2.ModelPredict.FromString,
          response_serializer=student__pb2.CommonGetResponse.SerializeToString,
      ),
      'GetExams': grpc.unary_unary_rpc_method_handler(
          servicer.GetExams,
          request_deserializer=student__pb2.CommonGetRequest.FromString,
          response_serializer=student__pb2.ExamResponse.SerializeToString,
      ),
      'GetPredicts': grpc.unary_unary_rpc_method_handler(
          servicer.GetPredicts,
          request_deserializer=student__pb2.GetPredictRequest.FromString,
          response_serializer=student__pb2.GetPredictResponse.SerializeToString,
      ),
      'StreamVideo': grpc.stream_stream_rpc_method_handler(
          servicer.StreamVideo,
          request_deserializer=student__pb2.StreamVideoRequest.FromString,
          response_serializer=student__pb2.CommonGetResponse.SerializeToString,
      ),
      'GetUserDetail': grpc.unary_unary_rpc_method_handler(
          servicer.GetUserDetail,
          request_deserializer=student__pb2.CommonGetRequest.FromString,
          response_serializer=student__pb2.CommonGetResponse.SerializeToString,
      ),
      'PutUserDetail': grpc.unary_unary_rpc_method_handler(
          servicer.PutUserDetail,
          request_deserializer=student__pb2.CommonGetRequest.FromString,
          response_serializer=student__pb2.CommonGetResponse.SerializeToString,
      ),
  }
  generic_handler = grpc.method_handlers_generic_handler(
      'student.StudentApp', rpc_method_handlers)
  server.add_generic_rpc_handlers((generic_handler,))
